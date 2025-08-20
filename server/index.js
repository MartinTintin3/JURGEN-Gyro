import express from 'express';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 8080;

// Ensure uploads dir exists
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) fs.mkdirSync(uploadsDir, { recursive: true });

// Multer storage config
const storage = multer.diskStorage({
  destination: (req, file, cb) => cb(null, uploadsDir),
  filename: (req, file, cb) => {
    // keep original name, prefix a timestamp
    const safe = file.originalname.replace(/[^a-zA-Z0-9_.-]/g, '_');
    cb(null, `${Date.now()}_${safe}`);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB
  fileFilter: (req, file, cb) => {
    const ok = /csv$/i.test(path.extname(file.originalname));
    ok ? cb(null, true) : cb(new Error('Only .csv files are allowed'));
  }
});

app.use(express.json());

// Serve uploaded files statically so the client can fetch them
app.use('/uploads', express.static(uploadsDir, { maxAge: '1h' }));

app.get("/api/available", (req, res) => {
	res.send({
		files: fs.readdirSync(uploadsDir),
	});
});

// API: upload a CSV -> returns a public URL
app.post('/api/upload', upload.single('file'), (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No file uploaded' });

  // quick mime sanity check
  if (!/text\/|csv|application\/vnd.ms-excel/.test(req.file.mimetype)) {
    // keep file but warn client
    console.warn('Suspicious mimetype:', req.file.mimetype);
  }

  const rel = `/uploads/${req.file.filename}`;
  return res.status(201).json({ url: rel, name: req.file.originalname });
});

// In production, serve the React build
const clientDist = path.join(__dirname, '..', 'client', 'dist');
if (fs.existsSync(clientDist)) {
  app.use(express.static(clientDist));
  app.get('*', (_, res) => res.sendFile(path.join(clientDist, 'index.html')));
}

app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ error: err.message || 'Server error' });
});

app.listen(PORT, () => console.log(`Server listening on http://localhost:${PORT}`));