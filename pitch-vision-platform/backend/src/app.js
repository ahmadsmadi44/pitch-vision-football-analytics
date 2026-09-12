import express from 'express';
import path from 'node:path';
import { readdir, readFile } from 'node:fs/promises';

const validId = value => /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$/.test(value);
const readJson = async file => JSON.parse(await readFile(file, 'utf8'));

export function createApp(dataDir, options = {}) {
  const app = express();
  const root = path.resolve(dataDir);
  const readRecord = options.readRecord || ((id, name) => readJson(path.join(root, id, `${name}.json`)));
  const tacticalRoot = path.resolve(root, '..', 'tactics');
  const readTactical = options.readTactical || (id => readJson(path.join(tacticalRoot, id, 'tactical.json')));
  app.disable('x-powered-by');
  app.use((req, res, next) => {
    res.set('X-Content-Type-Options', 'nosniff');
    next();
  });
  app.get('/api/health', (_req, res) => res.json({ status: 'ok' }));
  app.param('id', (req, res, next, id) => validId(id) ? next() : res.status(400).json({ error: 'Invalid match id' }));

  app.get('/api/tactics/:id', async (req, res) => res.json(await readTactical(req.params.id)));
  app.get('/api/tactics/:id/video.mp4', (req, res, next) => {
    if (options.mediaBaseUrl) return res.redirect(`${options.mediaBaseUrl}/tactics/${req.params.id}/video.mp4`);
    res.sendFile(path.join(tacticalRoot, req.params.id, 'video.mp4'), error => error && next(error));
  });
  app.get('/api/matches', async (_req, res) => {
    if (options.listMatches) return res.json(await options.listMatches());
    let entries;
    try { entries = await readdir(root, { withFileTypes: true }); }
    catch (error) { if (error.code === 'ENOENT') return res.json([]); throw error; }
    const matches = await Promise.all(entries.filter(e => e.isDirectory() && validId(e.name)).map(async e => {
      const meta = await readJson(path.join(root, e.name, 'meta.json'));
      return { ...meta, id: e.name };
    }));
    res.json(matches.sort((a, b) => a.id.localeCompare(b.id)));
  });
  app.get('/api/matches/:id', async (req, res) => {
    const dir = path.join(root, req.params.id);
    const [meta, ratings, heatmaps] = await Promise.all(['meta', 'ratings', 'heatmaps'].map(name => readRecord(req.params.id, name)));
    const stats = meta.hasStats ? await readRecord(req.params.id, 'stats') : {};
    const ids = [...new Set([...Object.keys(ratings), ...Object.keys(heatmaps)])];
    const players = ids.map(id => ({
      id, team: ratings[id]?.team ?? heatmaps[id]?.team ?? null,
      rating: ratings[id]?.rating_1_10 ?? null,
      rawStats: ratings[id]?.raw_stats ?? null,
      normalizedStats: ratings[id]?.normalized_stats ?? null,
      stats: stats[id] ?? null,
      heatmap: heatmaps[id] ?? null,
      thumbnail: meta.thumbnailIds?.includes(id) ? `/api/matches/${req.params.id}/thumbnails/${id}.jpg` : null,
    }));
    res.json({ ...meta, id: req.params.id, players });
  });
  app.get('/api/matches/:id/tracks', async (req, res) => res.json(await readRecord(req.params.id, 'tracks')));
  app.get('/api/matches/:id/video.mp4', (req, res, next) => {
    if (options.mediaBaseUrl) return res.redirect(`${options.mediaBaseUrl}/${req.params.id}/video.mp4`);
    res.sendFile(path.join(root, req.params.id, 'video.mp4'), error => error && next(error));
  });
  app.get('/api/matches/:id/thumbnails/:file', (req, res, next) => {
    if (!/^\d+\.jpg$/.test(req.params.file)) return res.status(400).json({ error: 'Invalid thumbnail' });
    if (options.mediaBaseUrl) return res.redirect(`${options.mediaBaseUrl}/${req.params.id}/thumbnails/${req.params.file}`);
    res.sendFile(path.join(root, req.params.id, 'thumbnails', req.params.file), error => error && next(error));
  });
  app.use((_req, res) => res.status(404).json({ error: 'Not found' }));
  app.use((error, _req, res, next) => {
    if (res.headersSent) return next(error);
    const missing = error.code === 'ENOENT' || error.status === 404;
    if (!missing) console.error(error);
    res.status(missing ? 404 : 500).json({ error: missing ? 'Match or asset not found' : 'Unable to load match data' });
  });
  return app;
}
