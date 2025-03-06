import React, { useState } from "react";
import { Link } from "react-router-dom";
import { useNavigate } from "react-router-dom";
import { logout } from "../../api/auth";

const Navbar = () => {
  const navigate = useNavigate();
  const isAuthenticated = !!localStorage.getItem("access_token");
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  const [showAdminMenu, setShowAdminMenu] = useState(false);

  const handleLogout = () => {
    logout();
    navigate("/auth/login");
  };

  const toggleAdminMenu = () => {
    setShowAdminMenu(!showAdminMenu);
  };

  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <Link to="/">Breast Cancer Detection</Link>
      </div>
      <div className="navbar-links">
        {isAuthenticated ? (
          <>
            <Link to="/user/home">Home</Link>
            <Link to="/predict">Predict</Link>

            {user.is_admin && (
              <div className="admin-dropdown">
                <button
                  className="admin-dropdown-button"
                  onClick={toggleAdminMenu}
                >
                  Admin ▼
                </button>
                {showAdminMenu && (
                  <div className="admin-dropdown-content">
                    <Link to="/admin/datasets">Datasets</Link>
                    <Link to="/admin/models">Models</Link>
                  </div>
                )}
              </div>
            )}
            <button onClick={handleLogout} className="nav-link-button">
              Logout
            </button>
          </>
        ) : (
          <>
            <Link to="/auth/login">Login</Link>
            <Link to="/auth/register">Register</Link>
          </>
        )}
      </div>
    </nav>
  );
};

export default Navbar;