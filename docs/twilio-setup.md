# Twilio + Cloudflare Tunnel Setup

## 1. Twilio Account
- Sign up at twilio.com
- Buy a phone number (~$1.15/mo)
- Note your Account SID and Auth Token from the dashboard

## 2. Environment Variables
Add to `.env`:
```
TWILIO_ACCOUNT_SID=ACxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxx
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx
TWILIO_WEBHOOK_BASE_URL=https://radio.yourdomain.com
```

## 3. Cloudflare Tunnel
Create a tunnel that routes to your local server:

```bash
cloudflared tunnel create radio-show
cloudflared tunnel route dns radio-show radio.yourdomain.com
```

Run during shows:
```bash
cloudflared tunnel --url http://localhost:8000 run radio-show
```

Or add to your NAS Cloudflare tunnel config.

## 4. Twilio Webhook Config
In the Twilio console, configure your phone number:
- Voice webhook URL: `https://radio.yourdomain.com/api/twilio/voice`
- Method: POST

## 5. Test
1. Start the server: `./run.sh`
2. Start the tunnel: `cloudflared tunnel run radio-show`
3. Call your Twilio number from a phone
4. You should see the caller appear in the queue panel
