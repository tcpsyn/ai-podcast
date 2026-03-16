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

    // Social crawler meta injection for episode pages
    if (url.pathname === "/episode.html" && url.searchParams.get("slug")) {
      const ua = (request.headers.get("User-Agent") || "").toLowerCase();
      const isCrawler = /facebookexternalhit|twitterbot|linkedinbot|slackbot|discordbot|telegrambot|whatsapp|pinterest|redditbot/i.test(ua);

      if (isCrawler) {
        const slug = url.searchParams.get("slug");
        try {
          const feedResp = await fetch("https://podcast.macneilmediagroup.com/@LukeAtTheRoost/feed.xml", {
            signal: AbortSignal.timeout(5000),
          });
          if (feedResp.ok) {
            const feedXml = await feedResp.text();
            const items = feedXml.split("<item>");
            let title = "";
            let description = "";

            for (let i = 1; i < items.length; i++) {
              const item = items[i];
              const linkMatch = item.match(/<link>(.*?)<\/link>/);
              if (linkMatch) {
                const itemSlug = linkMatch[1].split("/episodes/").pop()?.replace(/\/$/, "");
                if (itemSlug === slug) {
                  const titleMatch = item.match(/<title>(.*?)<\/title>/);
                  title = titleMatch ? titleMatch[1].replace(/<!\[CDATA\[|\]\]>/g, "").trim() : "";
                  const descMatch = item.match(/<description>([\s\S]*?)<\/description>/);
                  description = descMatch
                    ? descMatch[1].replace(/<!\[CDATA\[|\]\]>/g, "").replace(/<[^>]+>/g, "").trim().slice(0, 200)
                    : "";
                  break;
                }
              }
            }

            if (title) {
              const pageResp = await env.ASSETS.fetch(request);
              let html = await pageResp.text();

              const escTitle = title.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
              const escDesc = description.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
              const canonicalUrl = `https://lukeattheroost.com/episode.html?slug=${slug}`;

              html = html.replace(/<meta property="og:title"[^>]*>/, `<meta property="og:title" content="${escTitle}">`);
              html = html.replace(/<meta property="og:description"[^>]*>/, `<meta property="og:description" content="${escDesc}">`);
              html = html.replace(/<meta property="og:url"[^>]*>/, `<meta property="og:url" content="${canonicalUrl}">`);
              html = html.replace(/<meta name="twitter:title"[^>]*>/, `<meta name="twitter:title" content="${escTitle}">`);
              html = html.replace(/<meta name="twitter:description"[^>]*>/, `<meta name="twitter:description" content="${escDesc}">`);
              html = html.replace(/<title[^>]*>.*?<\/title>/, `<title>${escTitle} — Luke at the Roost</title>`);

              return new Response(html, {
                status: 200,
                headers: { "Content-Type": "text/html;charset=UTF-8" },
              });
            }
          }
        } catch (e) {
          // Fall through to static page
        }
      }
    }

    // All other requests — serve static assets
    return env.ASSETS.fetch(request);
  },
};
