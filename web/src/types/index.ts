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

export interface FaceInfo {
  face_id: string;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2] in original image pixels
  det_score: number;
  embedding: number[]; // 512-dim ArcFace embedding
  image_width: number; // original image width (for coordinate scaling)
  image_height: number; // original image height
}
