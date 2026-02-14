export interface SearchResult {
  id: number;
  image_url: string;
  event_name: string;
  event_year: number;
  album_title: string | null;
  flickr_photo_id: string | null;
  score: number;
}

export interface CropRect {
  x: number;
  y: number;
  w: number;
  h: number;
}
