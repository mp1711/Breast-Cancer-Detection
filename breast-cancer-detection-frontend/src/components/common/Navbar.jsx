import React, { useState, useEffect, useRef } from "react";
import { Link, useNavigate } from "react-router-dom";
import { logout } from "../../api/auth";

const Navbar = () => {
  const navigate = useNavigate();
  const isAuthenticated = !!localStorage.getItem("access_token");
  const user = JSON.parse(localStorage.getItem("user") || "{}");
  const [showAdminMenu, setShowAdminMenu] = useState(false);
  const dropdownRef = useRef(null);

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

  const imageStyle = {
    height: '50px',
    width: 'auto',
    objectFit: 'contain',
    maxWidth: '100%'
  };

  return (
    <>
      <nav style={{
        position: 'fixed',
        top: 0,
        width: '100%',
        backgroundColor: '#1e1e1e',
        zIndex: 1000
      }}>
        {/* Top row with images */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '0.5rem 1rem',
        }}>
          <img src="/top_left.jpg" alt="Top Left" style={imageStyle} />
          <img src="/center.jpg" alt="Center" style={imageStyle} />
          <img src="/top_right.jpg" alt="Top Right" style={imageStyle} />
        </div>

        {/* Bottom row with links */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          padding: '0.5rem 1rem',
          borderTop: '1px solid #333'
        }}>
          <div className="navbar-brand">
            <Link to="/" style={{ color: 'white', textDecoration: 'none' }}>
              Development of GUI Platform Towards Classification and Prediction of Breast Thermograms Based on Deep Learning Tools
            </Link>
          </div>
          <div className="navbar-links" style={{ display: 'flex', gap: '1rem' }}>
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
                <button onClick={handleLogout} className="nav-link-button">Logout</button>
              </>
            ) : (
              <>
                <Link to="/auth/login">Login</Link>
                <Link to="/auth/register">Register</Link>
              </>
            )}
          </div>
        </div>
      </nav>

      {/* Push content down below the fixed navbar (adjust height as needed) */}
      <div style={{ height: '140px' }}></div>
    </>
  );
};

export default Navbar;
