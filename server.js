// server.js — HF Router with model fallbacks + smarter response handling
import express from "express";
import dotenv from "dotenv";
import fetch from "node-fetch";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();
const app = express();
app.use(express.json({ limit: "2mb" }));

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
app.use(express.static(path.join(__dirname, "public")));

const HF_TOKEN = process.env.HUGGINGFACE_TOKEN;

// Try these in order; the first that works will be used.
// Feel free to reorder.
const MODELS = [
  "black-forest-labs/FLUX.1-schnell",              // fast, widely available
  "stabilityai/sdxl-turbo",                        // real-time style
  "stabilityai/stable-diffusion-2-1",              // classic SD 2.1
  "runwayml/stable-diffusion-v1-5"                 // SD 1.5
];

async function callModel(modelId, prompt, width, height) {
  const url = `https://router.huggingface.co/hf-inference/models/${modelId}`;
  const body = {
    inputs: prompt,
    parameters: {
      width,
      height,
      // reasonable defaults (some models ignore these)
      num_inference_steps: 4,
      guidance_scale: 0.0
    },
    options: { wait_for_model: true }
  };

  const resp = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${HF_TOKEN}`,
      "Content-Type": "application/json",
      "Accept": "image/png"
    },
    body: JSON.stringify(body)
  });

  if (!resp.ok) {
    const errTxt = await resp.text().catch(() => "");
    const e = new Error(`HF ${resp.status} for ${modelId}${errTxt ? `: ${errTxt}` : ""}`);
    e.status = resp.status;
    throw e;
  }

  // Some backends honor Accept:image/png and return bytes; others return JSON.
  const ct = resp.headers.get("content-type") || "";
  if (ct.includes("image/")) {
    const buf = await resp.arrayBuffer();
    return `data:image/png;base64,${Buffer.from(buf).toString("base64")}`;
  } else {
    // JSON path (rare). Try common fields:
    const data = await resp.json();
    // Look for base64 image fields if present
    const b64 =
      data?.b64_json ||
      data?.image_base64 ||
      (Array.isArray(data) && data[0]?.b64_json);
    if (!b64) {
      throw new Error(`HF response for ${modelId} was JSON, not image. Body: ${JSON.stringify(data).slice(0,300)}...`);
    }
    return `data:image/png;base64,${b64}`;
  }
}

async function generateOnce(prompt, size) {
  const [w, h] = String(size).split("x").map(Number);
  let firstErr;
  for (const m of MODELS) {
    try {
      // tweak defaults per model if you like
      return await callModel(m, prompt, w || 768, h || 768);
    } catch (e) {
      // Save the first error; continue trying the next model on 404/5xx
      if (!firstErr) firstErr = e;
      // If error is 401/403, no point continuing—auth issue
      if (e.status === 401 || e.status === 403) throw e;
      // For 404/5xx, try next model
    }
  }
  throw firstErr || new Error("All models failed.");
}

app.post("/api/generate", async (req, res) => {
  try {
    const { prompt, n = 1, size = "768x768" } = req.body || {};
    if (!prompt?.trim()) return res.status(400).json({ error: "Prompt is required." });
    if (!HF_TOKEN) return res.status(500).json({ error: "Missing HUGGINGFACE_TOKEN on server." });

    const count = Math.max(1, Math.min(4, Number(n) || 1));
    const tasks = Array.from({ length: count }, () => generateOnce(prompt, size));
    const images = await Promise.all(tasks);
    res.json({ images });
  } catch (err) {
    console.error(err);
    res.status(502).json({ error: String(err.message || err) });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`HF image generator running on http://localhost:${PORT}`);
});
