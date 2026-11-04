import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from './pages/Dashboard'; // AI Advisor page (student dashboard)
import FacultyDashboard from './pages/FacultyDashboard'; // Faculty Advisor page
import AdminDashboard from './pages/AdminDashboard'; // Admin Dashboard
import Profile from './pages/Profile'; // Profile page
import LoginPage from './pages/LoginPage';

const DEFAULT_USER = {
  id: 'demo-1',
  name: 'NIT Hamirpur Student',
  email: 'student@nith.ac.in',
  role: 'student',
  sex: 'Male',
  dateOfBirth: '2003-01-01'
};

const getStoredUser = () => {
  try {
    const user = JSON.parse(localStorage.getItem('user') || 'null');
    return user || DEFAULT_USER;
  } catch (_) {
    return DEFAULT_USER;
  }
};

const AppRouter = () => {
  const [user, setUser] = useState(getStoredUser);

  const handleLogin = (nextUser) => {
    setUser(nextUser);
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('role');
    localStorage.removeItem('user');
    setUser(DEFAULT_USER);
  };

  return (
    <Router>
      <Routes>
        <Route path="/login" element={<LoginPage onLogin={handleLogin} />} />

        {/* AI Advisor Route (default) */}
        <Route path="/" element={<Dashboard />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/student-dashboard" element={<Navigate to="/dashboard" replace />} />

        {/* Faculty Advisor Route - Direct open access */}
        <Route path="/faculty-dashboard" element={<FacultyDashboard />} />

        {/* Admin Dashboard Route - Direct open access */}
        <Route path="/admin-dashboard" element={<AdminDashboard />} />

        {/* Profile Route - Direct open access */}
        <Route path="/profile" element={<Profile onLogout={handleLogout} />} />

        {/* Catch-All Route */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Router>
  );
};

export default AppRouter;
