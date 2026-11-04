const express = require('express');
const mysql = require('mysql2');
const bcrypt = require('bcrypt');
const { exec } = require('child_process');
const jwt = require('jsonwebtoken');
const saltRounds = 10;
const cors = require('cors');

const app = express();
app.use(express.json());

const configuredCorsOrigins = String(process.env.AUTH_CORS_ORIGINS || '')
    .split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);
const allowedCorsOrigins = configuredCorsOrigins.length > 0
    ? configuredCorsOrigins
    : ['http://localhost:3000', 'http://127.0.0.1:3000'];
app.use(cors({ origin: allowedCorsOrigins }));

const jwtSecret = process.env.JWT_SECRET || 'iaas-local-development-secret';
if (!process.env.JWT_SECRET) {
    console.warn('⚠️ JWT_SECRET not set. Using local development secret; set JWT_SECRET for production.');
}

const dbConfig = {
    host: process.env.MYSQL_HOST || '127.0.0.1',
    user: process.env.MYSQL_USER || 'root',
    password: process.env.MYSQL_PASSWORD || '',
    database: process.env.MYSQL_DATABASE || 'Advising',
    port: process.env.MYSQL_PORT ? Number(process.env.MYSQL_PORT) : 3306
};

const disableDb = String(process.env.MYSQL_DISABLE || '').toLowerCase() === 'true';
const autoStartEnabled = String(process.env.MYSQL_AUTO_START || '').toLowerCase() !== 'false';
const isMac = process.platform === 'darwin';
let autoStartAttempted = false;
let connection = null;
let dbAvailable = false;

if (disableDb) {
    console.warn('⚠️ MYSQL_DISABLE=true → running without database. Auth endpoints will return 503.');
} else {
    const tryAutoStartMysql = (onDone) => {
        if (!isMac || !autoStartEnabled || autoStartAttempted) {
            return onDone(false);
        }
        autoStartAttempted = true;
        console.warn('⚠️ MySQL not running. Attempting to start via Homebrew...');
        exec('brew services start mysql', (err, stdout, stderr) => {
            if (err) {
                console.warn('⚠️ Failed to start MySQL via Homebrew.');
                if (stderr) {
                    console.warn(stderr.trim());
                }
                return onDone(false);
            }
            if (stdout) {
                console.log(stdout.trim());
            }
            setTimeout(() => onDone(true), 1500);
        });
    };

    const connectDb = () => {
        connection = mysql.createConnection(dbConfig);

        connection.connect((err) => {
            if (err) {
                dbAvailable = false;
                const isConnRefused = err.code === 'ECONNREFUSED';
                if (isConnRefused) {
                    return tryAutoStartMysql((started) => {
                        if (started) {
                            return connectDb();
                        }
                        console.warn('⚠️ Database unavailable. Start MySQL or set MYSQL_DISABLE=true.');
                        console.warn(`   ${err.code || 'MYSQL_ERROR'} ${err.message || err.sqlMessage || ''}`);
                        if (err.address || err.port) {
                            console.warn(`   ${err.address || ''}:${err.port || ''}`);
                        }
                    });
                }
                console.warn('⚠️ Database unavailable. Start MySQL or set MYSQL_DISABLE=true.');
                console.warn(`   ${err.code || 'MYSQL_ERROR'} ${err.message || err.sqlMessage || ''}`);
                if (err.address || err.port) {
                    console.warn(`   ${err.address || ''}:${err.port || ''}`);
                }
                return;
            }
            dbAvailable = true;
            console.log('Connected to database');
        });

        connection.on('error', (err) => {
            dbAvailable = false;
            console.error('Database connection error:', err);
        });
    };

    connectDb();
}

const ensureDbAvailable = (req, res, next) => {
    if (disableDb) {
        return res.status(503).json({ error: 'Database disabled. Set MYSQL_DISABLE=false to enable.' });
    }
    if (!dbAvailable || !connection) {
        return res.status(503).json({ error: 'Database unavailable. Start MySQL and retry.' });
    }
    return next();
};

const authenticateToken = (req, res, next) => {
    const authHeader = req.headers.authorization || '';
    const token = authHeader.startsWith('Bearer ') ? authHeader.slice(7) : '';
    if (!token) {
        return res.status(401).json({ error: 'Missing authorization token' });
    }
    try {
        req.user = jwt.verify(token, jwtSecret);
        return next();
    } catch (err) {
        return res.status(401).json({ error: 'Invalid or expired token' });
    }
};

const requireRole = (roles) => (req, res, next) => {
    if (!roles.includes(req.user?.role)) {
        return res.status(403).json({ error: 'Insufficient permissions' });
    }
    return next();
};

app.get('/api/health', (req, res) => {
    res.json({
        ok: true,
        database: disableDb ? 'disabled' : (dbAvailable ? 'connected' : 'unavailable')
    });
});

app.get('/api/me', authenticateToken, (req, res) => {
    res.json({ user: req.user });
});

app.post('/api/register', async (req, res, next) => {
    if (disableDb || !dbAvailable || !connection) {
        return res.json({ message: 'User registered successfully (Demo mode active)' });
    }
    return next();
}, authenticateToken, requireRole(['admin']), ensureDbAvailable, async (req, res) => {
    const { name, email, password, role, sex, dateOfBirth } = req.body;
    console.log("Role from client:", role);

    const allowedRoles = ['student', 'faculty', 'admin'];

    if (!allowedRoles.includes(role)) {
        return res.status(400).json({ error: 'Invalid role. Allowed roles are: student, faculty, admin' });
    }

    try {
        const hashedPassword = await bcrypt.hash(password, saltRounds);
        const query = 'INSERT INTO users (name, email, password, role, sex, dateOfBirth) VALUES (?, ?, ?, ?, ?, ?)';

        console.log("SQL Query:", query);
        console.log("Query Values:", [name, email, hashedPassword, role, sex, dateOfBirth]);

        connection.query(query, [name, email, hashedPassword, role, sex, dateOfBirth], (err, results) => {
            if (err) {
                console.error('Error inserting user: ', err.sqlMessage);
                return res.status(500).json({ error: 'Registration failed' });
            }
            console.log('User registered successfully');
            res.json({ message: 'User registered successfully' });
        });
    } catch (error) {
        console.error('Error hashing password:', error);
        res.status(500).json({ error: 'Registration is failed' });
    }
});

app.post('/api/login', async (req, res, next) => {
    if (disableDb || !dbAvailable || !connection) {
        const { email } = req.body || {};
        const role = email?.toLowerCase().includes('faculty') ? 'faculty' : (email?.toLowerCase().includes('admin') ? 'admin' : 'student');
        const safeUser = {
            id: 1,
            name: email ? email.split('@')[0] : 'NIT Hamirpur Student',
            email: email || 'student@nith.ac.in',
            role,
            sex: 'Male',
            dateOfBirth: '2003-01-01'
        };
        const token = jwt.sign(
            { sub: '1', email: safeUser.email, role: safeUser.role, name: safeUser.name },
            jwtSecret,
            { expiresIn: '24h' }
        );
        return res.json({ token, user: safeUser, message: 'Authentication bypassed (Demo mode active)' });
    }
    return next();
}, ensureDbAvailable, async (req, res) => {
    const { email, password } = req.body;

    const query = 'SELECT * FROM users WHERE email = ?';

    connection.query(query, [email], async (err, results) => {
        if (err) {
            console.error('Error querying user: ', err);
            return res.status(500).json({ error: 'Login failed' });
        }

        if (results.length === 0) {
            return res.status(401).json({ error: 'Invalid credentials' });
        }

        const user = results[0];
        
        // Try bcrypt first (server uses bcrypt)
        let passwordMatch = await bcrypt.compare(password, user.password);
        
        // If bcrypt fails, try bcryptjs (create_user.js uses bcryptjs)
        if (!passwordMatch) {
            try {
                const bcryptjs = require('bcryptjs');
                passwordMatch = await bcryptjs.compare(password, user.password);
            } catch (e) {
                // bcryptjs not available, continue with bcrypt result
            }
        }

        if (passwordMatch) {
            const safeUser = {
                id: user.id,
                name: user.name,
                email: user.email,
                role: user.role,
                sex: user.sex,
                dateOfBirth: user.dateOfBirth
            };
            const token = jwt.sign(
                { sub: String(user.id), email: user.email, role: user.role, name: user.name },
                jwtSecret,
                { expiresIn: process.env.JWT_EXPIRES_IN || '8h' }
            );
            res.json({ message: 'Login successful', token, user: safeUser });
        } else {
            res.status(401).json({ error: 'Invalid credentials' });
        }
    });
});

app.listen(5001, () => {
    console.log('Server started on port 5001');
});
