'use client';

import { useState } from 'react';
import GameIcon from './GameIcon';
import { useProgress } from './ProgressProvider';
import { MAX_PROGRESS_BYTES, readProgress } from '@/lib/progressSync';

export default function SyncPanel() {
  const sync = useProgress();
  const [key, setKey] = useState('');
  const [connecting, setConnecting] = useState(false);
  const [notice, setNotice] = useState('');
  const busy = connecting || sync.status === 'syncing';
  const label = sync.status === 'syncing' ? 'Syncing…' : sync.status === 'synced' ? 'Up to date'
    : sync.status === 'pending' ? 'Sync pending' : sync.status === 'error' ? 'Needs attention'
    : sync.status === 'unavailable' ? 'Setup needed' : 'This device only';

  function downloadBackup() {
    const url = URL.createObjectURL(new Blob([JSON.stringify(sync.progress)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url; link.download = `counting-progress-${sync.today || 'backup'}.json`;
    link.click(); window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return <section className="sync-panel" aria-label="Device sync">
    <div className="section-top"><div><span className="eyebrow">Your private profile</span><h2>Pick up on any device.</h2></div>
      <span className={`sync-status sync-status-${sync.status}`} role="status"><GameIcon name={sync.status === 'synced' ? 'check' : 'cloud'} />{label}</span></div>
    <p>Keep your daily streak, personal best, and saved Quick Play, guided counting, and full-test results together.</p>
    {sync.linked ? <>
      <p className="sync-detail">This device is linked. Enter the same private key on your other devices.</p>
      <div className="button-row"><button className="primary-button" disabled={busy} onClick={() => { setNotice(''); void sync.sync(); }}>Sync now</button>
        <button className="quiet-button" onClick={async () => {
          try { await navigator.clipboard.writeText(sync.getKey()); setNotice('Sync key copied. Paste it on your other device.'); }
          catch { setNotice('Clipboard access is unavailable. Use the key you saved during setup.'); }
        }}>Copy sync key</button>
        <button className="text-button" onClick={() => { sync.disconnect(); setNotice('Disconnected here. Your local progress and cloud copy are kept.'); }}>Disconnect this device</button></div>
      {sync.lastSynced && <p className="sync-detail">Last synced {new Date(sync.lastSynced).toLocaleString()}.</p>}
    </> : sync.status === 'unavailable' ? <div><p className="sync-detail">Cloud sync needs a one-time connection to storage. You can keep playing locally.</p>
      <button className="quiet-button" onClick={() => window.location.reload()}>Check setup again</button></div> :
      <form className="sync-connect" onSubmit={async event => {
        event.preventDefault(); if (connecting) return;
        setConnecting(true); setNotice('');
        try { if (await sync.connect(key.trim())) { setKey(''); setNotice('Device linked. Existing progress has been merged.'); } }
        finally { setConnecting(false); }
      }}><label htmlFor="sync-key">Private sync key</label><div className="sync-key-row"><input id="sync-key" type="password" autoComplete="current-password" spellCheck={false}
          minLength={32} maxLength={256} required value={key} onChange={event => setKey(event.target.value)} placeholder="Paste your key" />
        <button className="primary-button" disabled={!sync.loaded || busy || key.trim().length < 32}>{connecting ? 'Connecting…' : 'Connect device'}</button></div>
        <p className="sync-detail">Enter it once per browser. Anyone with this key can access the same profile, so keep it private.</p></form>}
    {sync.message && <p className="sync-message" role="status">{sync.message}</p>}
    {notice && <p className="sync-message" role="status">{notice}</p>}
    <div className="sync-bottom"><p>Daily streak timezone: <strong>{sync.progress.timeZone.replaceAll('_', ' ')}</strong><br />If your connection drops while playing, saved results retry when you reconnect.</p>
      <details><summary>Backup &amp; restore</summary><div className="button-row"><button className="quiet-button" disabled={!sync.loaded} onClick={downloadBackup}>Download backup</button>
        <label className="quiet-button sync-import">Merge a backup<input type="file" accept="application/json,.json" onChange={async event => {
          const file = event.target.files?.[0]; event.target.value = ''; if (!file) return;
          try {
            if (file.size > MAX_PROGRESS_BYTES) throw new Error('The backup is too large.');
            const data = readProgress(await file.text()); sync.importProgress(data); setNotice('Backup merged with the progress already here.');
          } catch { setNotice('This backup could not be read. Your existing progress has been kept.'); }
        }} /></label></div></details></div>
  </section>;
}
