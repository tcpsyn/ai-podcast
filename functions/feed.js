export async function onRequest() {
  const feedUrl = 'https://podcast.macneilmediagroup.com/@LukeAtTheRoost/feed.xml';
  const res = await fetch(feedUrl);
  const xml = await res.text();
  return new Response(xml, {
    headers: {
      'Content-Type': 'application/xml',
      'Access-Control-Allow-Origin': '*',
      'Cache-Control': 'public, max-age=300',
    },
  });
}
