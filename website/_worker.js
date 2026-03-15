const VOICEMAIL_XML = `<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say voice="woman">Luke at the Roost is off the air right now. Leave a message after the beep and we may play it on the next show!</Say>
    <Record maxLength="120" action="https://radioshow.macneilmediagroup.com/api/signalwire/voicemail-complete" playBeep="true" />
    <Say voice="woman">Thank you for calling. Goodbye!</Say>
    <Hangup/>
</Response>`;

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/api/signalwire/voice") {
      try {
        const body = await request.text();
        const resp = await fetch("https://radioshow.macneilmediagroup.com/api/signalwire/voice", {
          method: "POST",
          headers: { "Content-Type": "application/x-www-form-urlencoded" },
          body: body,
          signal: AbortSignal.timeout(5000),
        });

        if (resp.ok) {
          return new Response(await resp.text(), {
            status: 200,
            headers: { "Content-Type": "application/xml" },
          });
        }
      } catch (e) {
        // Server unreachable or timed out
      }

      return new Response(VOICEMAIL_XML, {
        status: 200,
        headers: { "Content-Type": "application/xml" },
      });
    }

    // RSS feed proxy
    if (url.pathname === "/feed") {
      try {
        const resp = await fetch("https://podcast.macneilmediagroup.com/@LukeAtTheRoost/feed.xml", {
          signal: AbortSignal.timeout(8000),
        });
        if (resp.ok) {
          return new Response(await resp.text(), {
            status: 200,
            headers: {
              "Content-Type": "application/xml",
              "Access-Control-Allow-Origin": "*",
              "Cache-Control": "public, max-age=300",
            },
          });
        }
      } catch (e) {
        // Castopod unreachable
      }
      return new Response("Feed unavailable", { status: 502 });
    }

    // Plausible analytics proxy (bypass ad blockers)
    if (url.pathname === "/p/script") {
      const resp = await fetch("https://plausible.macneilmediagroup.com/js/script.file-downloads.hash.outbound-links.pageview-props.revenue.tagged-events.js");
      return new Response(await resp.text(), {
        headers: {
          "Content-Type": "application/javascript",
          "Cache-Control": "public, max-age=86400",
        },
      });
    }

    if (url.pathname === "/p/event" && request.method === "POST") {
      const body = await request.text();
      const resp = await fetch("https://plausible.macneilmediagroup.com/api/event", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "User-Agent": request.headers.get("User-Agent") || "",
          "X-Forwarded-For": request.headers.get("CF-Connecting-IP") || request.headers.get("X-Forwarded-For") || "",
        },
        body,
      });
      return new Response(resp.body, {
        status: resp.status,
        headers: { "Content-Type": resp.headers.get("Content-Type") || "text/plain" },
      });
    }

    // All other requests — serve static assets
    return env.ASSETS.fetch(request);
  },
};
