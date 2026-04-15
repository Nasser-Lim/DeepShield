export type Verdict = "safe" | "caution" | "risk";

export interface ModelScore {
  score: number;
  heatmap_b64: string | null;
}

export interface InferResponse {
  effort: ModelScore;
  xray: ModelScore;
  spsl: ModelScore;
  face_bbox: [number, number, number, number];
  overlay_b64: string;
}

export interface AnalyzeResult {
  id: string;
  verdict: Verdict;
  final_score: number;
  scores: InferResponse;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

export async function analyzeImage(file: File): Promise<AnalyzeResult> {
  const form = new FormData();
  form.append("image", file);

  const res = await fetch(`${API_URL}/analyze`, { method: "POST", body: form });
  if (!res.ok) {
    let message: string;
    try {
      const body = await res.json();
      message = body?.detail ?? body?.message ?? JSON.stringify(body);
    } catch {
      message = await res.text();
    }
    throw new Error(message);
  }
  return (await res.json()) as AnalyzeResult;
}
