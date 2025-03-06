import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { login } from "../../api/auth";
import Alert from "../../components/common/Alert";

const LoginPage = () => {
  const [username, setUserName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await login({ username, password });

      // Store token from the API response
      localStorage.setItem("access_token", response.access_token);

      // Store user data that now comes directly from the API response
      const userData = {
        username: response.username,
        email: response.email,
        is_admin: response.is_admin,
      };
      localStorage.setItem("user", JSON.stringify(userData));

      // Redirect based on admin status
      const redirect = response.is_admin ? "/admin/datasets" : "/user/home";
      navigate(redirect);
    } catch (err) {
      setError(err.response?.data?.detail || "Login failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <h2>Login</h2>
      {error && <Alert message={error} type="error" />}
      <form onSubmit={handleSubmit}>
        <div>
          <label>User Name:</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUserName(e.target.value)}
            required
          />
        </div>
        <div>
          <label>Password:</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <button type="submit" disabled={loading}>
          {loading ? "Logging in..." : "Login"}
        </button>
      </form>
    </div>
  );
};

export default LoginPage;
