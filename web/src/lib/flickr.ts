/** Swap the size suffix in a Flickr static URL.
 *
 * Size suffixes: s=75sq, q=150sq, t=100, m=240, z=640, b=1024, h=1600, k=2048
 */
const FLICKR_SIZE_RE = /(_[a-z0-9])\.jpg$/i;

export function flickrUrlResize(url: string, size: string = "z"): string {
  return url.replace(FLICKR_SIZE_RE, `_${size}.jpg`);
}
