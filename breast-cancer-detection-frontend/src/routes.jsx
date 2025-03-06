import React from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import LoginPage from "./pages/auth/LoginPage";
import RegisterPage from "./pages/auth/RegisterPage";
import HomePage from "./pages/user/HomePage";
import PredictPage from "./pages/user/PredictPage";
import DatasetsPage from "./pages/admin/DatasetsPage";
import ModelsPage from "./pages/admin/ModelsPage";

const isAuthenticated = () => {
  const token = localStorage.getItem("access_token");
  if (!token) return false;

  try {
    const payload = JSON.parse(atob(token.split(".")[1]));
    return payload.exp * 1000 > Date.now();
  } catch (e) {
    console.error("Error parsing token:", e);
    return false;
  }
};

const isAdmin = () => {
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  return user.is_admin === true;
};

const PrivateRoute = ({ children }) => {
  return isAuthenticated() ? children : <Navigate to="/auth/login" />;
};

const AdminRoute = ({ children }) => {
  return isAuthenticated() ? (
    isAdmin() ? (
      children
    ) : (
      <Navigate to="/user/home" />
    )
  ) : (
    <Navigate to="/auth/login" />
  );
};

const AppRoutes = () => {
  return (
    <Routes>
      <Route path="/auth/login" element={<LoginPage />} />
      <Route path="/auth/register" element={<RegisterPage />} />
      <Route
        path="/user/home"
        element={
          <PrivateRoute>
            <HomePage />
          </PrivateRoute>
        }
      />
      <Route
        path="/predict"
        element={
          <PrivateRoute>
            <PredictPage />
          </PrivateRoute>
        }
      />
      
      <Route
        path="/admin/datasets"
        element={
          <AdminRoute>
            <DatasetsPage />
          </AdminRoute>
        }
      />
      <Route
        path="/admin/models"
        element={
          <AdminRoute>
            <ModelsPage />
          </AdminRoute>
        }
      />
      <Route path="/" element={<Navigate to="/user/home" />} />
    </Routes>
  );
};

export default AppRoutes;