import { createProgressHandler } from '../../../lib/progressServer';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const revalidate = 0;

const handleProgress = createProgressHandler();
export const GET = handleProgress;
export const POST = handleProgress;
