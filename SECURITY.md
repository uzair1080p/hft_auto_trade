# 🔐 Security Configuration

## Dashboard Authentication

The HFT Trading Dashboard now includes secure authentication to protect sensitive trading information.

### Default Credentials
- **Username**: `admin`
- **Password**: `trading123`

### Changing Credentials

#### Option 1: Environment Variables (Recommended)
Set these environment variables before starting the system:

```bash
export DASHBOARD_USERNAME="your_secure_username"
export DASHBOARD_PASSWORD="your_secure_password"
```

#### Option 2: Docker Compose Environment
Add to your `.env` file:
```
DASHBOARD_USERNAME=your_secure_username
DASHBOARD_PASSWORD=your_secure_password
```

#### Option 3: Direct Code Modification
Edit `ui_dashboard_fixed.py` and change:
```python
DEFAULT_USERNAME = os.getenv('DASHBOARD_USERNAME', 'your_secure_username')
DEFAULT_PASSWORD = os.getenv('DASHBOARD_PASSWORD', 'your_secure_password')
```

### Security Best Practices

1. **Use Strong Passwords**: Choose complex passwords with uppercase, lowercase, numbers, and special characters
2. **Change Default Credentials**: Always change the default username and password
3. **Environment Variables**: Use environment variables instead of hardcoding credentials
4. **HTTPS**: When deploying publicly, always use HTTPS
5. **Network Security**: Consider using VPN or IP whitelisting for additional security
6. **Regular Updates**: Keep the system and dependencies updated
7. **Access Logging**: Monitor dashboard access logs
8. **Session Management**: Logout when not using the dashboard

### Production Deployment Security

When deploying to production:

1. **Use HTTPS**: Configure SSL/TLS certificates
2. **Reverse Proxy**: Use nginx or similar for additional security layers
3. **Firewall**: Configure firewall rules to restrict access
4. **VPN**: Consider VPN-only access for sensitive environments
5. **Monitoring**: Set up alerts for failed login attempts
6. **Backup**: Regularly backup configuration and data

### Example Production Setup

```bash
# Set secure credentials
export DASHBOARD_USERNAME="trading_admin_2024"
export DASHBOARD_PASSWORD="K9#mP2$vL8@nQ4!xR7"

# Start with HTTPS (example with nginx)
docker-compose up -d
```

### Security Features

- ✅ Password hashing with SHA-256
- ✅ Session-based authentication
- ✅ Secure logout functionality
- ✅ Environment variable support
- ✅ No hardcoded credentials in production
- ✅ Session timeout (browser-based)

### Warning

⚠️ **Never use default credentials in production!** Always change the username and password before deploying publicly.
