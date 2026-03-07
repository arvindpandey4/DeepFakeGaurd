import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, X, Check, AlertTriangle, Play, Shield, ShieldAlert, Cpu, Activity, Clock, Database, ChevronRight, DownloadCloud, RefreshCw, Trash2 } from 'lucide-react';

// API Configuration
const API_URL = "http://localhost:8000";

function App() {
    const [file, setFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [demoVideos, setDemoVideos] = useState([]);
    const [availableRemoteVideos, setAvailableRemoteVideos] = useState([]);
    const [isDownloading, setIsDownloading] = useState(false);
    const [showDownloadModal, setShowDownloadModal] = useState(false);
    const [downloadCategory, setDownloadCategory] = useState('REAL');
    const [cacheReady, setCacheReady] = useState(false);
    const [remoteVideoPage, setRemoteVideoPage] = useState({ REAL: 0, DEEPFAKE: 0 });
    const pollTimerRef = useRef(null);
    const fileInputRef = useRef(null);
    const PAGE_SIZE = 5;

    // Fetch demo videos on mount
    useEffect(() => {
        fetchDemoVideos();
    }, []);

    const fetchDemoVideos = async () => {
        try {
            const res = await axios.get(`${API_URL}/demo-videos`);
            setDemoVideos(res.data);
        } catch (err) {
            console.error("Failed to load demo videos", err);
        }
    };

    const fetchRemoteVideos = async () => {
        setIsDownloading("fetching");
        try {
            const res = await axios.get(`${API_URL}/available-remote-videos`);
            setAvailableRemoteVideos(res.data.videos || []);
            setCacheReady(res.data.is_ready);

            if (!res.data.is_ready) {
                // Still loading — poll every 3s
                if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
                pollTimerRef.current = setTimeout(fetchRemoteVideos, 3000);
            } else {
                if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
            }
        } catch (err) {
            console.error("Failed to load remote videos", err);
            setError("Failed to fetch dataset from Hugging Face.");
        } finally {
            setIsDownloading(false);
        }
    };

    const handleDownload = async (video) => {
        if (video.is_downloaded) return;

        setIsDownloading(video.id);
        try {
            await axios.post(`${API_URL}/download-remote-video`, {
                video_id: video.id
            });
            // Refresh counts
            await fetchRemoteVideos();
            await fetchDemoVideos();
        } catch (err) {
            console.error("Download failed", err);
            setError("Failed to download video.");
        } finally {
            setIsDownloading(false);
        }
    };

    const handleFileChange = (e) => {
        const selectedFile = e.target.files[0];
        if (selectedFile) {
            setFile(selectedFile);
            setPreviewUrl(URL.createObjectURL(selectedFile));
            setResult(null);
            setError(null);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        const selectedFile = e.dataTransfer.files[0];
        if (selectedFile && selectedFile.type.startsWith('video/')) {
            setFile(selectedFile);
            setPreviewUrl(URL.createObjectURL(selectedFile));
            setResult(null);
            setError(null);
        }
    };

    const startAnalysis = async () => {
        if (!file) return;

        setIsAnalyzing(true);
        setError(null);
        setResult(null);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await axios.post(`${API_URL}/analyze`, formData, {
                headers: { "Content-Type": "multipart/form-data" },
            });
            // Delay for UI effect
            setTimeout(() => {
                setResult(response.data);
                setIsAnalyzing(false);
            }, 1500);
        } catch (err) {
            console.error(err);
            setError("Analysis failed. Please try again.");
            setIsAnalyzing(false);
        }
    };

    const runDemoVideo = async (video) => {
        // Set UI state to analyzing immediately
        setFile(null); // Clear manual file if any
        setPreviewUrl(null); // Will be set after response or if we had URL in backend response
        setResult(null);
        setError(null);
        setIsAnalyzing(true);

        try {
            const response = await axios.post(`${API_URL}/analyze-demo`, {
                path_id: video.path_id
            });

            setTimeout(() => {
                // Set the video URL from the response so it plays in the main window
                setPreviewUrl(`${API_URL}${response.data.video_url}`);
                setResult(response.data);
                setIsAnalyzing(false);
                // Also fake a file object so the UI shows the video player
                setFile({ name: video.filename });
            }, 1000);

        } catch (err) {
            console.error(err);
            setError("Demo analysis failed.");
            setIsAnalyzing(false);
        }
    }

    const reset = () => {
        setFile(null);
        setPreviewUrl(null);
        setResult(null);
        setError(null);
    };

    const clearCache = async () => {
        if (!window.confirm("Are you sure you want to clear all downloaded videos and analysis results? This cannot be undone.")) {
            return;
        }

        try {
            await axios.post(`${API_URL}/clear-cache`);
            // Reset local state
            reset();
            setDemoVideos([]);
            setAvailableRemoteVideos([]);
            // Refresh video lists
            await fetchDemoVideos();
            alert("Cache cleared successfully!");
        } catch (err) {
            console.error("Failed to clear cache", err);
            setError("Failed to clear cache.");
        }
    };

    return (
        <div className="min-h-screen bg-neutral-950 text-white selection:bg-cyan-500/30 font-sans relative overflow-hidden">
            {/* Background Gradients */}
            <div className="fixed top-0 left-0 w-full h-full pointer-events-none z-0">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[120px]" />
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/10 rounded-full blur-[120px]" />
            </div>

            <div className="relative z-10 max-w-7xl mx-auto px-6 py-12">

                {/* Header */}
                <header className="flex justify-between items-center mb-12">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
                            <Shield className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h1 className="text-2xl font-display font-bold tracking-tight">Deepfake<span className="text-cyan-400">Guard</span></h1>
                            <p className="text-xs text-neutral-400 uppercase tracking-widest font-medium">Adaptive Inference System</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-4">
                        <button
                            onClick={clearCache}
                            className="p-2.5 rounded-full bg-red-500/10 border border-red-500/20 text-red-500 hover:bg-red-500/20 transition-all group relative"
                            title="Clear Cache & Results"
                        >
                            <Trash2 className="w-5 h-5" />
                            <span className="absolute -bottom-10 right-0 bg-red-900 border border-red-500/30 text-[10px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                                Clear Cache
                            </span>
                        </button>
                        <div className="flex bg-white/5 rounded-full p-1 border border-white/5 backdrop-blur-md">
                            <button
                                onClick={() => {
                                    setDownloadCategory('REAL');
                                    setShowDownloadModal(true);
                                    fetchRemoteVideos();
                                }}
                                className="px-4 py-1.5 rounded-full text-xs font-bold transition-all hover:bg-emerald-500/20 text-emerald-400 flex items-center gap-2"
                            >
                                <Check className="w-3.5 h-3.5" />
                                Ingest Real
                            </button>
                            <div className="w-px h-4 bg-white/10 self-center mx-1" />
                            <button
                                onClick={() => {
                                    setDownloadCategory('DEEPFAKE');
                                    setShowDownloadModal(true);
                                    fetchRemoteVideos();
                                }}
                                className="px-4 py-1.5 rounded-full text-xs font-bold transition-all hover:bg-red-500/20 text-red-400 flex items-center gap-2"
                            >
                                <ShieldAlert className="w-3.5 h-3.5" />
                                Ingest Deepfake
                            </button>
                        </div>
                        <div className="text-sm font-medium text-neutral-500 bg-white/5 px-4 py-2 rounded-full border border-white/5 backdrop-blur-md">
                            v1.0.0 • MesoNet-4
                        </div>
                    </div>
                </header>

                {/* Main Content */}
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

                    {/* Left Column: Upload & Video */}
                    <div className="lg:col-span-8 space-y-6">

                        {/* Video / Upload Area */}
                        <div
                            className={`relative aspect-video rounded-3xl overflow-hidden border transition-all duration-500 group ${file ? 'border-white/10 bg-black' : 'border-white/10 bg-white/5 border-dashed hover:border-cyan-500/50 hover:bg-white/10'
                                }`}
                            onDragOver={(e) => e.preventDefault()}
                            onDrop={handleDrop}
                        >
                            {previewUrl || (isAnalyzing && !previewUrl) ? (
                                <div className="relative w-full h-full bg-black flex items-center justify-center">
                                    {previewUrl && (
                                        <video
                                            src={previewUrl}
                                            className="w-full h-full object-contain"
                                            controls
                                            autoPlay
                                            loop
                                            muted
                                        />
                                    )}

                                    {/* Scanning Overlay */}
                                    {isAnalyzing && (
                                        <motion.div
                                            initial={{ opacity: 0 }}
                                            animate={{ opacity: 1 }}
                                            className="absolute inset-0 z-20 pointer-events-none bg-cyan-500/10"
                                        >
                                            <div className="absolute top-0 left-0 w-full h-1 bg-cyan-400 shadow-[0_0_20px_rgba(34,211,238,0.8)] animate-scan" />
                                            <div className="absolute top-4 left-4 font-mono text-xs text-cyan-400 bg-black/50 px-2 py-1 rounded border border-cyan-500/30">
                                                SCANNING FRAMES...
                                            </div>
                                            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20" />
                                        </motion.div>
                                    )}

                                    {/* Loading State without URL */}
                                    {isAnalyzing && !previewUrl && (
                                        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center">
                                            <div className="w-16 h-16 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin mb-4" />
                                            <p className="text-cyan-400 font-mono text-sm">Loading Demo Video...</p>
                                        </div>
                                    )}

                                    <button
                                        onClick={reset}
                                        className="absolute top-4 right-4 p-2 bg-black/60 text-white rounded-full hover:bg-black/80 transition-colors z-30"
                                    >
                                        <X className="w-5 h-5" />
                                    </button>
                                </div>
                            ) : (
                                <div
                                    className="absolute inset-0 flex flex-col items-center justify-center cursor-pointer pointer-events-auto"
                                    onClick={() => fileInputRef.current?.click()}
                                >
                                    <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-6 border border-white/10 group-hover:scale-110 transition-transform duration-300">
                                        <Upload className="w-8 h-8 text-neutral-400 group-hover:text-cyan-400 transition-colors" />
                                    </div>
                                    <h3 className="text-xl font-medium text-neutral-200 mb-2">Upload Video</h3>
                                    <p className="text-neutral-500 text-sm max-w-xs text-center">Drag & drop or click to browse. Supported formats: MP4, MOV, AVI.</p>
                                    <input
                                        type="file"
                                        ref={fileInputRef}
                                        onChange={handleFileChange}
                                        className="hidden"
                                        accept="video/*"
                                    />
                                </div>
                            )}
                        </div>

                        {/* Controls */}
                        {file && !result && !isAnalyzing && (
                            <motion.div
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="flex justify-center"
                            >
                                <button
                                    onClick={startAnalysis}
                                    className="bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white px-8 py-4 rounded-full font-medium text-lg shadow-lg shadow-cyan-500/25 transition-all hover:scale-105 active:scale-95 flex items-center gap-3"
                                >
                                    <Activity className="w-5 h-5" />
                                    Start Analysis
                                </button>
                            </motion.div>
                        )}

                        {/* Error Message */}
                        {error && (
                            <div className="bg-red-500/10 border border-red-500/20 text-red-500 px-4 py-3 rounded-xl flex items-center gap-3">
                                <AlertTriangle className="w-5 h-5" />
                                {error}
                            </div>
                        )}

                        {/* Pipeline Visualization (Always visible, active when Result exists) */}
                        <AnimatePresence mode="wait">
                            {result && (
                                <motion.div
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="bg-white/5 border border-white/10 rounded-2xl p-6"
                                >
                                    <div className="flex justify-between items-center mb-6">
                                        <h3 className="text-sm font-medium text-neutral-400 uppercase tracking-wider">Pipeline Execution Path</h3>
                                        <div className="text-xs font-mono text-neutral-500">
                                            ID: {result.filename}
                                        </div>
                                    </div>

                                    <div className="space-y-6 relative">
                                        <div className="absolute left-[19px] top-2 bottom-4 w-0.5 bg-white/10 z-0" />

                                        <PipelineStep
                                            number="1"
                                            status="completed"
                                            data={result.stage_results}
                                            isExit={result.exit_stage === 1}
                                        />
                                        <PipelineStep
                                            number="2"
                                            status={result.exit_stage >= 2 ? "completed" : "skipped"}
                                            isExit={result.exit_stage === 2}
                                        />
                                        <PipelineStep
                                            number="3"
                                            status={result.exit_stage >= 3 ? "completed" : "skipped"}
                                            isExit={result.exit_stage === 3}
                                        />
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                    </div>

                    {/* Right Column: Library & Results */}
                    <div className="lg:col-span-4 space-y-6">

                        {/* Result Card (Shows on Top if Result exists) */}
                        <AnimatePresence>
                            {result && (
                                <motion.div
                                    initial={{ opacity: 0, x: 20 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    className="space-y-6"
                                >
                                    {/* Final Verdict Card */}
                                    <div className={`p-8 rounded-3xl border backdrop-blur-xl relative overflow-hidden ${result.label === 'DEEPFAKE'
                                        ? 'bg-red-500/10 border-red-500/30 shadow-lg shadow-red-900/20'
                                        : 'bg-emerald-500/10 border-emerald-500/30 shadow-lg shadow-emerald-900/20'
                                        }`}>
                                        <div className="relative z-10 text-center">
                                            <div className="flex justify-center mb-4">
                                                {result.label === 'DEEPFAKE' ? (
                                                    <div className="p-4 rounded-full bg-red-500/20 border border-red-500/50">
                                                        <ShieldAlert className="w-10 h-10 text-red-500" />
                                                    </div>
                                                ) : (
                                                    <div className="p-4 rounded-full bg-emerald-500/20 border border-emerald-500/50">
                                                        <Shield className="w-10 h-10 text-emerald-500" />
                                                    </div>
                                                )}
                                            </div>
                                            <h2 className="text-4xl font-display font-bold mb-2 tracking-tight">
                                                {result.label}
                                            </h2>
                                            <p className={`text-sm font-medium tracking-wide uppercase ${result.label === 'DEEPFAKE' ? 'text-red-400' : 'text-emerald-400'}`}>
                                                Confidence: {(result.confidence * 100).toFixed(1)}%
                                            </p>
                                        </div>
                                        <div className={`absolute -top-20 -right-20 w-64 h-64 rounded-full blur-[100px] opacity-40 z-0 ${result.label === 'DEEPFAKE' ? 'bg-red-600' : 'bg-emerald-600'
                                            }`} />
                                    </div>

                                    {/* Details Grid */}
                                    <div className="grid grid-cols-2 gap-3">
                                        <DetailCard
                                            icon={<Activity className="w-4 h-4 text-cyan-400" />}
                                            label="Exit Stage"
                                            value={`Stage ${result.exit_stage}`}
                                        />
                                        <DetailCard
                                            icon={<Clock className="w-4 h-4 text-purple-400" />}
                                            label="Time"
                                            value={`${result.total_time.toFixed(2)}s`}
                                        />
                                        <DetailCard
                                            icon={<Check className="w-4 h-4 text-blue-400" />}
                                            label="Prob"
                                            value={result.probability.toFixed(3)}
                                        />
                                        <DetailCard
                                            icon={<Upload className="w-4 h-4 text-orange-400" />}
                                            label="Eff."
                                            value={result.exit_stage < 3 ? "High" : "Std"}
                                        />
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* Demo Library — tabbed + paginated */}
                        <DataIngestionPanel
                            demoVideos={demoVideos}
                            isAnalyzing={isAnalyzing}
                            runDemoVideo={runDemoVideo}
                        />
                    </div>

                </div>
            </div>

            {/* Download Modal */}
            <AnimatePresence>
                {showDownloadModal && (
                    <div className="fixed inset-0 z-[100] flex items-center justify-center p-6">
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setShowDownloadModal(false)}
                            className="absolute inset-0 bg-black/80 backdrop-blur-md"
                        />
                        <motion.div
                            initial={{ scale: 0.9, opacity: 0, y: 20 }}
                            animate={{ scale: 1, opacity: 1, y: 0 }}
                            exit={{ scale: 0.9, opacity: 0, y: 20 }}
                            className="relative w-full max-w-lg bg-neutral-900 border border-white/10 rounded-3xl overflow-hidden overflow-y-auto max-h-[80vh] shadow-2xl"
                        >
                            <div className="p-6 border-b border-white/5 flex justify-between items-center sticky top-0 bg-neutral-900 z-10">
                                <div>
                                    <div className="flex items-center gap-2">
                                        <h3 className="text-lg font-bold">Dataset Engine</h3>
                                        {isDownloading === "fetching" && (
                                            <RefreshCw className="w-4 h-4 text-cyan-500 animate-spin" />
                                        )}
                                    </div>
                                    <p className="text-xs text-neutral-400">Using Hugging Face Datasets</p>
                                </div>
                                <div className="flex items-center gap-2">
                                    <button
                                        onClick={async () => {
                                            try { await axios.post(`${API_URL}/sync-remote-videos`); } catch (_) { }
                                            setCacheReady(false);
                                            setAvailableRemoteVideos([]);
                                            fetchRemoteVideos();
                                        }}
                                        disabled={isDownloading === "fetching"}
                                        className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-white/5 border border-white/10 text-xs font-bold hover:bg-white/10 transition-all text-cyan-400"
                                    >
                                        <RefreshCw className={`w-3.5 h-3.5 ${isDownloading === "fetching" ? "animate-spin" : ""}`} />
                                        Sync Repository
                                    </button>
                                    <button
                                        onClick={() => setShowDownloadModal(false)}
                                        className="p-2 hover:bg-white/5 rounded-full transition-colors"
                                    >
                                        <X className="w-5 h-5 text-neutral-400" />
                                    </button>
                                </div>
                            </div>

                            {/* Tabs */}
                            <div className="flex border-b border-white/5">
                                <button
                                    onClick={() => { setDownloadCategory('REAL'); setRemoteVideoPage(p => ({ ...p, REAL: 0 })); }}
                                    className={`flex-1 py-3 text-xs font-bold uppercase tracking-wider transition-all border-b-2 ${downloadCategory === 'REAL'
                                        ? 'text-emerald-400 border-emerald-500 bg-emerald-500/5'
                                        : 'text-neutral-500 border-transparent hover:text-neutral-300'
                                        }`}
                                >
                                    Normal Videos (Real)
                                </button>
                                <button
                                    onClick={() => { setDownloadCategory('DEEPFAKE'); setRemoteVideoPage(p => ({ ...p, DEEPFAKE: 0 })); }}
                                    className={`flex-1 py-3 text-xs font-bold uppercase tracking-wider transition-all border-b-2 ${downloadCategory === 'DEEPFAKE'
                                        ? 'text-red-400 border-red-500 bg-red-500/5'
                                        : 'text-neutral-500 border-transparent hover:text-neutral-300'
                                        }`}
                                >
                                    Deepfake Samples
                                </button>
                            </div>


                            {/* Paginated + animated video list */}
                            {(() => {
                                const filtered = availableRemoteVideos.filter(v => v.type === downloadCategory);
                                const currentPage = remoteVideoPage[downloadCategory];
                                const pageVideos = filtered.slice(currentPage * PAGE_SIZE, (currentPage + 1) * PAGE_SIZE);
                                const hasNext = (currentPage + 1) * PAGE_SIZE < filtered.length;
                                const hasPrev = currentPage > 0;
                                return (
                                    <>
                                        <div className="p-6 space-y-3" style={{ minHeight: '220px' }}>
                                            {filtered.length === 0 ? (
                                                <div className="flex flex-col items-center py-12 text-center">
                                                    <RefreshCw className="w-8 h-8 text-cyan-500 animate-spin mb-4" />
                                                    <p className="text-sm text-neutral-400">
                                                        {cacheReady ? 'No videos found.' : 'Loading from Hugging Face...'}
                                                    </p>
                                                    {!cacheReady && (
                                                        <p className="text-xs text-neutral-600 mt-1">This takes ~30s on first load</p>
                                                    )}
                                                </div>
                                            ) : (
                                                <AnimatePresence mode="wait">
                                                    <motion.div
                                                        key={`${downloadCategory}-pg-${currentPage}`}
                                                        initial={{ opacity: 0, x: 20 }}
                                                        animate={{ opacity: 1, x: 0 }}
                                                        exit={{ opacity: 0, x: -20 }}
                                                        transition={{ duration: 0.18, ease: 'easeInOut' }}
                                                        className="space-y-3"
                                                    >
                                                        {pageVideos.map((video) => (
                                                            <div
                                                                key={video.id}
                                                                className="p-4 bg-white/5 border border-white/5 rounded-2xl flex items-center justify-between gap-4 group hover:bg-white/10 transition-all"
                                                            >
                                                                <div className="min-w-0">
                                                                    <div className="flex items-center gap-2 mb-1">
                                                                        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${video.type === 'DEEPFAKE' ? 'bg-red-500/20 text-red-400' : 'bg-emerald-500/20 text-emerald-400'}`}>
                                                                            {video.type}
                                                                        </span>
                                                                        <h4 className="font-medium text-sm truncate">{video.name}</h4>
                                                                    </div>
                                                                    <p className="text-xs text-neutral-500 line-clamp-1">{video.description}</p>
                                                                </div>
                                                                <button
                                                                    onClick={() => handleDownload(video)}
                                                                    disabled={video.is_downloaded || isDownloading === video.id}
                                                                    className={`flex-shrink-0 px-4 py-2 rounded-xl text-xs font-bold transition-all ${video.is_downloaded ? 'bg-emerald-500/10 text-emerald-500 cursor-default'
                                                                        : isDownloading === video.id ? 'bg-cyan-500/20 text-cyan-400 cursor-wait'
                                                                            : 'bg-cyan-500 text-black hover:bg-cyan-400 active:scale-95'}`}
                                                                >
                                                                    {video.is_downloaded ? (
                                                                        <span className="flex items-center gap-1"><Check className="w-3 h-3" /> Ready</span>
                                                                    ) : isDownloading === video.id ? (
                                                                        <span className="flex items-center gap-2"><RefreshCw className="w-3 h-3 animate-spin" /> Saving...</span>
                                                                    ) : 'Download'}
                                                                </button>
                                                            </div>
                                                        ))}
                                                    </motion.div>
                                                </AnimatePresence>
                                            )}
                                        </div>

                                        {/* Pagination nav */}
                                        {filtered.length > PAGE_SIZE && (
                                            <div className="px-6 pb-5 flex items-center justify-between gap-3 border-t border-white/5 pt-4">
                                                <button
                                                    onClick={() => setRemoteVideoPage(p => ({ ...p, [downloadCategory]: p[downloadCategory] - 1 }))}
                                                    disabled={!hasPrev}
                                                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold border transition-all ${hasPrev ? 'border-white/10 text-neutral-300 hover:bg-white/10 hover:border-white/20 active:scale-95'
                                                        : 'border-white/5 text-neutral-600 cursor-not-allowed opacity-40'}`}
                                                >
                                                    <ChevronRight className="w-3.5 h-3.5 rotate-180" /> Back
                                                </button>
                                                <span className="text-[10px] font-mono text-neutral-400">
                                                    {currentPage * PAGE_SIZE + 1}–{Math.min((currentPage + 1) * PAGE_SIZE, filtered.length)}
                                                    <span className="text-neutral-600"> / {filtered.length} videos</span>
                                                </span>
                                                <button
                                                    onClick={() => setRemoteVideoPage(p => ({ ...p, [downloadCategory]: p[downloadCategory] + 1 }))}
                                                    disabled={!hasNext}
                                                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold border transition-all ${hasNext
                                                        ? `border-white/10 hover:bg-white/10 hover:border-white/20 active:scale-95 ${downloadCategory === 'DEEPFAKE' ? 'text-red-400' : 'text-emerald-400'}`
                                                        : 'border-white/5 text-neutral-600 cursor-not-allowed opacity-40'}`}
                                                >
                                                    Load More <ChevronRight className="w-3.5 h-3.5" />
                                                </button>
                                            </div>
                                        )}
                                    </>
                                );
                            })()}

                            <div className="p-6 bg-white/5 border-t border-white/5">
                                <p className="text-[10px] text-neutral-500 text-center italic">
                                    Note: Downloads might take 10-30 seconds depending on file size and connection.
                                </p>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </div>
    );
}

function DataIngestionPanel({ demoVideos, isAnalyzing, runDemoVideo }) {
    const [demoTab, setDemoTab] = useState('REAL');
    const [demoPage, setDemoPage] = useState({ REAL: 0, DEEPFAKE: 0 });
    const DEMO_PAGE = 5;

    const filtered = demoVideos.filter(v => v.type === demoTab);
    const currentPage = demoPage[demoTab];
    const pageVideos = filtered.slice(currentPage * DEMO_PAGE, (currentPage + 1) * DEMO_PAGE);
    const hasNext = (currentPage + 1) * DEMO_PAGE < filtered.length;
    const hasPrev = currentPage > 0;

    return (
        <div className="bg-neutral-900/50 border border-white/5 rounded-3xl overflow-hidden backdrop-blur-sm">
            {/* Header */}
            <div className="p-5 border-b border-white/5 flex justify-between items-center">
                <h3 className="font-medium text-white flex items-center gap-2">
                    <Database className="w-4 h-4 text-cyan-400" />
                    Data Ingestion
                </h3>
                <div className="flex gap-2 text-[10px] text-neutral-500 font-mono">
                    <span className="flex items-center gap-1"><div className="w-1.5 h-1.5 rounded-full bg-emerald-500" /> {demoVideos.filter(v => v.type === 'REAL').length}R</span>
                    <span className="flex items-center gap-1"><div className="w-1.5 h-1.5 rounded-full bg-red-500" /> {demoVideos.filter(v => v.type === 'DEEPFAKE').length}F</span>
                </div>
            </div>

            {demoVideos.length === 0 ? (
                <div className="p-10 text-center text-neutral-500 text-sm">
                    <AlertTriangle className="w-8 h-8 mx-auto mb-2 opacity-30" />
                    No videos found in cache.<br />Download from the Dataset Engine first.
                </div>
            ) : (
                <>
                    {/* REAL / DEEPFAKE tabs */}
                    <div className="flex border-b border-white/5">
                        <button
                            onClick={() => { setDemoTab('REAL'); setDemoPage(p => ({ ...p, REAL: 0 })); }}
                            className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider transition-all border-b-2 ${demoTab === 'REAL' ? 'text-emerald-400 border-emerald-500 bg-emerald-500/5' : 'text-neutral-500 border-transparent hover:text-neutral-300'}`}
                        >
                            Real ({demoVideos.filter(v => v.type === 'REAL').length})
                        </button>
                        <button
                            onClick={() => { setDemoTab('DEEPFAKE'); setDemoPage(p => ({ ...p, DEEPFAKE: 0 })); }}
                            className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider transition-all border-b-2 ${demoTab === 'DEEPFAKE' ? 'text-red-400 border-red-500 bg-red-500/5' : 'text-neutral-500 border-transparent hover:text-neutral-300'}`}
                        >
                            Deepfake ({demoVideos.filter(v => v.type === 'DEEPFAKE').length})
                        </button>
                    </div>

                    {/* Video list */}
                    <div className="p-2 space-y-1" style={{ minHeight: '180px' }}>
                        {filtered.length === 0 ? (
                            <div className="p-8 text-center text-neutral-500 text-sm">
                                <AlertTriangle className="w-6 h-6 mx-auto mb-2 opacity-30" />
                                No {demoTab.toLowerCase()} videos in cache.
                            </div>
                        ) : (
                            <AnimatePresence mode="wait">
                                <motion.div
                                    key={`demo-${demoTab}-${currentPage}`}
                                    initial={{ opacity: 0, x: 16 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    exit={{ opacity: 0, x: -16 }}
                                    transition={{ duration: 0.16, ease: 'easeInOut' }}
                                    className="space-y-1"
                                >
                                    {pageVideos.map((video, idx) => (
                                        <button
                                            key={video.path_id || idx}
                                            onClick={() => runDemoVideo(video)}
                                            disabled={isAnalyzing}
                                            className="w-full text-left p-3 rounded-xl hover:bg-white/5 transition-all group flex items-center justify-between border border-transparent hover:border-white/5"
                                        >
                                            <div className="flex items-center gap-3">
                                                <div className={`w-7 h-7 rounded-lg flex items-center justify-center font-mono text-xs flex-shrink-0 ${demoTab === 'DEEPFAKE' ? 'bg-red-500/10 text-red-400' : 'bg-emerald-500/10 text-emerald-400'}`}>
                                                    {currentPage * DEMO_PAGE + idx + 1}
                                                </div>
                                                <div className="min-w-0">
                                                    <div className="text-sm font-medium text-neutral-200 truncate pr-2 group-hover:text-cyan-400 transition-colors">
                                                        {video.filename}
                                                    </div>
                                                    <div className="text-[10px] text-neutral-500 uppercase">
                                                        {video.type} Source
                                                    </div>
                                                </div>
                                            </div>
                                            <ChevronRight className="w-4 h-4 text-neutral-600 group-hover:text-white transition-colors flex-shrink-0" />
                                        </button>
                                    ))}
                                </motion.div>
                            </AnimatePresence>
                        )}
                    </div>

                    {/* Pagination nav */}
                    {filtered.length > DEMO_PAGE ? (
                        <div className="px-3 py-3 flex items-center justify-between gap-2 border-t border-white/5">
                            <button
                                onClick={() => setDemoPage(p => ({ ...p, [demoTab]: p[demoTab] - 1 }))}
                                disabled={!hasPrev}
                                className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-bold border transition-all ${hasPrev ? 'border-white/10 text-neutral-300 hover:bg-white/10 active:scale-95' : 'border-white/5 text-neutral-600 cursor-not-allowed opacity-40'}`}
                            >
                                <ChevronRight className="w-3 h-3 rotate-180" /> Back
                            </button>
                            <span className="text-[10px] font-mono text-neutral-500">
                                {currentPage * DEMO_PAGE + 1}–{Math.min((currentPage + 1) * DEMO_PAGE, filtered.length)}
                                <span className="text-neutral-600"> / {filtered.length}</span>
                            </span>
                            <button
                                onClick={() => setDemoPage(p => ({ ...p, [demoTab]: p[demoTab] + 1 }))}
                                disabled={!hasNext}
                                className={`flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-bold border transition-all ${hasNext
                                    ? `border-white/10 hover:bg-white/10 active:scale-95 ${demoTab === 'DEEPFAKE' ? 'text-red-400' : 'text-emerald-400'}`
                                    : 'border-white/5 text-neutral-600 cursor-not-allowed opacity-40'}`}
                            >
                                More <ChevronRight className="w-3 h-3" />
                            </button>
                        </div>
                    ) : (
                        <div className="p-3 bg-white/5 border-t border-white/5 text-center">
                            <p className="text-[10px] text-neutral-500">Click any video to run instant analysis</p>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}

function DetailCard({ icon, label, value }) {
    return (
        <div className="bg-white/5 border border-white/10 p-3 rounded-2xl">
            <div className="flex items-center gap-2 mb-1">
                {icon}
                <span className="text-[10px] text-neutral-400 uppercase font-medium">{label}</span>
            </div>
            <div className="text-lg font-display font-medium text-white">{value}</div>
        </div>
    )
}

function PipelineStep({ number, status, isExit }) {
    const isCompleted = status === 'completed';
    const isSkipped = status === 'skipped';

    return (
        <div className={`relative z-10 flex items-start gap-4 ${isSkipped ? 'opacity-30' : 'opacity-100'}`}>
            <div className={`w-10 h-10 rounded-full flex items-center justify-center font-mono text-sm border-2 transition-colors ${isCompleted
                ? isExit ? 'bg-cyan-500 border-cyan-500 text-black shadow-[0_0_15px_rgba(6,182,212,0.5)]' : 'bg-neutral-800 border-cyan-500/50 text-cyan-400'
                : 'bg-neutral-900 border-neutral-700 text-neutral-500'
                }`}>
                {status === 'completed' && !isExit ? <Check className="w-4 h-4" /> : number}
            </div>
            <div className="pt-2">
                <h4 className={`text-sm font-medium ${isCompleted ? 'text-white' : 'text-neutral-500'}`}>
                    Stage {number}
                    {isExit && <span className="ml-2 text-[10px] bg-cyan-500/20 text-cyan-300 px-2 py-0.5 rounded border border-cyan-500/30 uppercase">Final Decision</span>}
                </h4>
                <p className="text-xs text-neutral-500 mt-0.5">
                    {number === '1' && "Fast Inference • 128px • 0.5 fps"}
                    {number === '2' && "Balanced Inference • 192px • 2 fps"}
                    {number === '3' && "Accurate Inference • 256px • 5 fps"}
                </p>
            </div>
        </div>
    )
}

export default App;
