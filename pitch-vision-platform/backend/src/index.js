import { fileURLToPath } from 'node:url';
import { createApp } from './app.js';

// The default is relative to this file, independent of the terminal's working folder.
const resolved = process.env.DATA_DIR || fileURLToPath(new URL('../../data/matches', import.meta.url));
const port = Number(process.env.PORT || 3001);
createApp(resolved).listen(port, '127.0.0.1', () => console.log(`Pitch Vision API: http://localhost:${port}`));
