import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { loginUser } from '../services/apiService';

const LoginPage = ({ onLogin = () => {} }) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const navigate = useNavigate();

    const handleQuickLogin = (role, defaultName, defaultEmail) => {
        const user = {
            id: `demo-${role}`,
            name: defaultName,
            email: defaultEmail,
            role,
            sex: 'Male',
            dateOfBirth: '2003-01-01'
        };
        onLogin(user);
        localStorage.setItem('token', 'demo-token');
        localStorage.setItem('role', role);
        localStorage.setItem('user', JSON.stringify(user));

        if (role === 'faculty') navigate('/faculty-dashboard');
        else if (role === 'admin') navigate('/admin-dashboard');
        else navigate('/dashboard');
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        try {
            const response = await loginUser({ email, password });
            const user = response.user;
            onLogin(user);
            localStorage.setItem('token', response.token);
            localStorage.setItem('role', user.role);
            localStorage.setItem('user', JSON.stringify(user));

            if (user.role === 'student') navigate('/dashboard');
            else if (user.role === 'faculty') navigate('/faculty-dashboard');
            else if (user.role === 'admin') navigate('/admin-dashboard');
            else navigate('/dashboard');
        } catch (err) {
            setError(err.message || 'Login failed');
        }
    };

    return (
        <div className="login-container">
            <h2>NIT Hamirpur Academic Advisor</h2>
            <p style={{ textAlign: 'center', color: '#666', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
                Open Demo Mode — Direct access enabled (No password required)
            </p>
            {error && <p className="error">{error}</p>}
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', marginBottom: '1.5rem' }}>
                <button
                    type="button"
                    style={{ background: '#2563eb', color: '#fff', padding: '0.75rem', border: 'none', borderRadius: '6px', cursor: 'pointer', fontWeight: 600 }}
                    onClick={() => handleQuickLogin('student', 'NIT Hamirpur Student', 'student@nith.ac.in')}
                >
                    🎓 Enter as Student (Leo Chat)
                </button>
                <button
                    type="button"
                    style={{ background: '#475569', color: '#fff', padding: '0.6rem', border: 'none', borderRadius: '6px', cursor: 'pointer' }}
                    onClick={() => handleQuickLogin('faculty', 'Faculty Member', 'faculty@nith.ac.in')}
                >
                    👨‍🏫 Enter as Faculty (Course Graphs)
                </button>
                <button
                    type="button"
                    style={{ background: '#334155', color: '#fff', padding: '0.6rem', border: 'none', borderRadius: '6px', cursor: 'pointer' }}
                    onClick={() => handleQuickLogin('admin', 'Administrator', 'admin@nith.ac.in')}
                >
                    ⚙️ Enter as Admin
                </button>
            </div>

            <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: '1rem', marginTop: '0.5rem' }}>
                <p style={{ fontSize: '0.8rem', color: '#888', marginBottom: '0.5rem', textAlign: 'center' }}>Or sign in with custom credentials:</p>
                <form onSubmit={handleSubmit}>
                    <input type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
                    <input type="password" placeholder="Password (optional in demo)" value={password} onChange={(e) => setPassword(e.target.value)} />
                    <button type="submit">Login</button>
                </form>
            </div>
        </div>
    );
};

export default LoginPage;
