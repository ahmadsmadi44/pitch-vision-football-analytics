import serverless from 'serverless-http';
import { S3Client, GetObjectCommand } from '@aws-sdk/client-s3';
import { createApp } from './app.js';

const s3 = new S3Client({});
async function read(key) {
  try {
    const result = await s3.send(new GetObjectCommand({ Bucket: process.env.DATA_BUCKET, Key: key }));
    return JSON.parse(await result.Body.transformToString());
  } catch (error) {
    if (error.name === 'NoSuchKey') error.code = 'ENOENT';
    throw error;
  }
}
export const handler = serverless(createApp('/tmp', {
  readRecord: (id, name) => read(`matches/${id}/${name}.json`),
  readTactical: id => read(`tactics/${id}/tactical.json`),
  listMatches: () => read('matches/index.json'),
  mediaBaseUrl: process.env.MEDIA_BASE_URL,
}));
