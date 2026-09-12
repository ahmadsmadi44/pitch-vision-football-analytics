import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { once } from 'node:events';
import { createApp } from '../src/app.js';

test('match API preserves unrated tracks, rejects traversal, and supports video ranges', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'pitch-vision-api-'));
  const dir = path.join(root, 'demo');
  await mkdir(dir);
  await Promise.all([
    writeFile(path.join(dir, 'meta.json'), JSON.stringify({ title: 'Fixture', thumbnailIds: [], hasStats: true })),
    writeFile(path.join(dir, 'ratings.json'), JSON.stringify({ 1: { team: 1, rating_1_10: 7 } })),
    writeFile(path.join(dir, 'heatmaps.json'), JSON.stringify({ 1: { team: 1 }, 2: { team: 'referee' } })),
    writeFile(path.join(dir, 'stats.json'), JSON.stringify({ 1: { observed_seconds: 2.4, peak_speed_kmh: 18 } })),
    writeFile(path.join(dir, 'video.mp4'), Buffer.from('0123456789')),
  ]);
  const server = createApp(root).listen(0, '127.0.0.1');
  await once(server, 'listening');
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    const matches = await (await fetch(`${base}/api/matches`)).json();
    assert.equal(matches.length, 1);
    const match = await (await fetch(`${base}/api/matches/demo`)).json();
    assert.equal(match.players.length, 2);
    assert.equal(match.players.find(p => p.id === '1').stats.observed_seconds, 2.4);
    assert.equal(match.players.find(p => p.id === '2').stats, null);
    assert.equal(match.players.find(p => p.id === '2').rating, null);
    assert.equal((await fetch(`${base}/api/matches/missing`)).status, 404);
    assert.equal((await fetch(`${base}/api/matches/bad%2Fid`)).status, 400);
    const video = await fetch(`${base}/api/matches/demo/video.mp4`, { headers: { Range: 'bytes=2-5' } });
    assert.equal(video.status, 206);
    assert.equal(await video.text(), '2345');
    assert.equal(video.headers.get('content-range'), 'bytes 2-5/10');
  } finally {
    await new Promise(resolve => server.close(resolve));
    await rm(root, { recursive: true, force: true });
  }
});

test('tactical API keeps source data separate and supports scrubbing', async () => {
  const baseDir = await mkdtemp(path.join(os.tmpdir(), 'football-tactics-api-'));
  const matches = path.join(baseDir, 'matches');
  const tactics = path.join(baseDir, 'tactics', 'topview-demo');
  await mkdir(matches); await mkdir(tactics, {recursive:true});
  await writeFile(path.join(tactics,'tactical.json'),JSON.stringify({source:'model',frames:[{time:0,ball:null}]}));
  await writeFile(path.join(tactics,'video.mp4'),Buffer.from('0123456789'));
  const server=createApp(matches).listen(0,'127.0.0.1');await once(server,'listening');
  const base=`http://127.0.0.1:${server.address().port}`;
  try {
    assert.deepEqual(await (await fetch(`${base}/api/matches`)).json(),[]);
    const result=await (await fetch(`${base}/api/tactics/topview-demo`)).json();
    assert.equal(result.source,'model');assert.equal(result.frames[0].ball,null);
    assert.equal((await fetch(`${base}/api/tactics/missing`)).status,404);
    assert.equal((await fetch(`${base}/api/tactics/bad%2Fid`)).status,400);
    const video=await fetch(`${base}/api/tactics/topview-demo/video.mp4`,{headers:{Range:'bytes=3-6'}});
    assert.equal(video.status,206);assert.equal(await video.text(),'3456');
  } finally {await new Promise(resolve=>server.close(resolve));await rm(baseDir,{recursive:true,force:true});}
});

test('tactical API supports object-storage data and media', async () => {
  const root = await mkdtemp(path.join(os.tmpdir(), 'football-tactics-cloud-'));
  const server = createApp(root, {
    readTactical: async id => ({ id, source: 'object-storage' }),
    mediaBaseUrl: 'https://media.example.test',
  }).listen(0, '127.0.0.1');
  await once(server, 'listening');
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    const tactical = await (await fetch(`${base}/api/tactics/topview-demo`)).json();
    assert.equal(tactical.source, 'object-storage');
    const media = await fetch(`${base}/api/tactics/topview-demo/video.mp4`, { redirect: 'manual' });
    assert.equal(media.status, 302);
    assert.equal(media.headers.get('location'), 'https://media.example.test/tactics/topview-demo/video.mp4');
  } finally {
    await new Promise(resolve => server.close(resolve));
    await rm(root, { recursive: true, force: true });
  }
});
