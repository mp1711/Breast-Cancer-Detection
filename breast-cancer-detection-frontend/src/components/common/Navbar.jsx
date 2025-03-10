import React, { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { useNavigate } from "react-router-dom";
import { logout } from "../../api/auth";

const Navbar = () => {
  const navigate = useNavigate();
  const isAuthenticated = !!localStorage.getItem("access_token");
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  const [showAdminMenu, setShowAdminMenu] = useState(false);
  const dropdownRef = useRef(null); // Reference for dropdown

  const handleLogout = () => {
    logout();
    navigate("/auth/login");
  };

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setShowAdminMenu(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

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
              <div className="admin-dropdown" ref={dropdownRef}>
                <button
                  className="admin-dropdown-button"
                  onClick={() => setShowAdminMenu((prev) => !prev)}
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
