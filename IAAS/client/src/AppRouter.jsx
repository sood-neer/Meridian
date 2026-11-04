import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Dashboard from './pages/Dashboard'; // AI Advisor page (student dashboard)
import FacultyDashboard from './pages/FacultyDashboard'; // Faculty Advisor page
import AdminDashboard from './pages/AdminDashboard'; // Admin Dashboard
import Profile from './pages/Profile'; // Profile page
import LoginPage from './pages/LoginPage';

const getStoredUser = () => {
  try {
    return JSON.parse(localStorage.getItem('user') || 'null');
  } catch (_) {
    return null;
  }
};

const RequireAuth = ({ user, roles, children }) => {
  if (!user) {
    return <Navigate to="/login" replace />;
  }
  if (roles?.length && !roles.includes(user.role)) {
    return <Navigate to="/dashboard" replace />;
  }
  return children;
};

const AppRouter = () => {
  const [user, setUser] = useState(getStoredUser);

  const handleLogin = (nextUser) => {
    setUser(nextUser);
  };

  const handleLogout = () => {
    localStorage.clear();
    setUser(null);
  };

  return (
    <Router>
      <Routes>
        <Route path="/login" element={<LoginPage onLogin={handleLogin} />} />

        {/* AI Advisor Route (default) */}
        <Route path="/" element={<Dashboard />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/student-dashboard" element={<Navigate to="/dashboard" replace />} />

        {/* Faculty Advisor Route */}
        <Route
          path="/faculty-dashboard"
          element={
            <RequireAuth user={user} roles={['faculty', 'admin']}>
              <FacultyDashboard />
            </RequireAuth>
          }
        />

        {/* Admin Dashboard Route */}
        <Route
          path="/admin-dashboard"
          element={
            <RequireAuth user={user} roles={['admin']}>
              <AdminDashboard />
            </RequireAuth>
          }
        />

        {/* Profile Route */}
        <Route
          path="/profile"
          element={
            <RequireAuth user={user}>
              <Profile onLogout={handleLogout} />
            </RequireAuth>
          }
        />

        {/* Catch-All Route */}
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </Router>
  );
};

export default AppRouter;
