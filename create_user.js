const bcryptjs = require('bcryptjs');
const mysql = require('mysql');

const connection = mysql.createConnection({
    host: 'localhost',
    user: 'advisor_user',
    password: 'Test1234!@#$',
    database: 'Advising'
});

const email = 'jod@psu.edu';
const password = '12345';

bcryptjs.hash(password, 10, (err, hash) => {
    if (err) {
        console.error('Error hashing password:', err);
        process.exit(1);
    }
    
    // First check if user exists
    connection.query('SELECT id FROM users WHERE email = ?', [email], (err, results) => {
        if (err) {
            console.error('Error checking user:', err);
            connection.end();
            process.exit(1);
        }
        
        if (results.length > 0) {
            // Update existing user
            connection.query('UPDATE users SET password = ?, name = ?, role = ? WHERE email = ?', 
                [hash, 'Jod', 'student', email], (err, results) => {
                if (err) {
                    console.error('Error updating user:', err);
                    connection.end();
                    process.exit(1);
                }
                console.log('✅ User updated successfully!');
                console.log('Email: ' + email);
                console.log('Password: ' + password);
                connection.end();
            });
        } else {
            // Create new user
            const query = 'INSERT INTO users (name, email, password, role, sex, dateOfBirth) VALUES (?, ?, ?, ?, ?, ?)';
            connection.query(query, ['Jod', email, hash, 'student', 'Male', '2000-01-01'], (err, results) => {
                if (err) {
                    console.error('Error creating user:', err);
                    connection.end();
                    process.exit(1);
                }
                console.log('✅ User created successfully!');
                console.log('Email: ' + email);
                console.log('Password: ' + password);
                connection.end();
            });
        }
    });
});





