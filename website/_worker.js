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

    // Umami analytics proxy (bypass ad blockers)
    if (url.pathname === "/api/send" && request.method === "POST") {
      const body = await request.text();
      const resp = await fetch("https://plausible.macneilmediagroup.com/api/send", {
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

    if (url.pathname === "/p/script") {
      const resp = await fetch("https://plausible.macneilmediagroup.com/script.js");
      return new Response(await resp.text(), {
        headers: {
          "Content-Type": "application/javascript",
          "Cache-Control": "public, max-age=86400",
        },
      });
    }

    if (url.pathname === "/p/event" && request.method === "POST") {
      const body = await request.text();
      const resp = await fetch("https://plausible.macneilmediagroup.com/api/send", {
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

    // Legacy query-param episode URLs -> clean static paths.
    //
    // This replaces the old user-agent-gated meta injection, which rewrote
    // <title>/og: tags only for a hardcoded list of social crawlers. Googlebot
    // was never on that list, so search engines only ever saw the generic
    // shell — and serving crawlers different HTML than users is cloaking.
    // Episode pages are static now, so nothing needs UA sniffing.
    //
    // Published YouTube descriptions and social posts still point at the old
    // form, so this redirect has to stay indefinitely.
    if (url.pathname === "/episode.html" && url.searchParams.get("slug")) {
      // The slug lands in a Location header. Restricting it to the character
      // set real slugs use prevents CRLF injection and open-redirect via a
      // crafted ?slug= (e.g. "//evil.com" or a value containing %0d%0a).
      const slug = url.searchParams.get("slug").replace(/[^a-z0-9-]/gi, "");
      if (slug) {
        return Response.redirect(`https://lukeattheroost.com/episode/${slug}/`, 301);
      }
    }

    // All other requests — serve static assets
    return env.ASSETS.fetch(request);
  },
};
