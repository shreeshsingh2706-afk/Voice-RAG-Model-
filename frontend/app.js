document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const micBtn = document.getElementById("mic-btn");
    const recordingStatus = document.getElementById("recording-status");
    const recordingTimer = document.getElementById("recording-timer");
    const queryForm = document.getElementById("query-form");
    const queryInput = document.getElementById("query-input");
    const relevanceBadge = document.getElementById("relevance-badge");
    const detectedLangBadge = document.getElementById("detected-lang-badge");
    const transcriptContainer = document.getElementById("transcript-container");
    const transcriptText = document.getElementById("transcript-text");
    const answerDisplay = document.getElementById("answer-text-display");
    const loader = document.getElementById("loader");
    const loaderStatus = document.getElementById("loader-status");
    const sourcesCount = document.getElementById("sources-count");
    const sourcesList = document.getElementById("sources-list");
    
    // Latency Elements
    const latencyStt = document.getElementById("latency-stt");
    const latencyRetrieval = document.getElementById("latency-retrieval");
    const latencyGen = document.getElementById("latency-gen");
    const latencyTotal = document.getElementById("latency-total");
    const engineSelect = document.getElementById("engine-select");
    const retrievalLabel = document.getElementById("retrieval-label");
    
    // Recorder State variables
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;
    let timerInterval = null;
    let secondsElapsed = 0;

    // Check host to lock down engine selection in cloud deployment
    const hostname = window.location.hostname.toLowerCase();
    const isLocal = 
        hostname === "localhost" || 
        hostname === "127.0.0.1" || 
        hostname === "" || 
        hostname.startsWith("192.168.") || 
        hostname.startsWith("10.") || 
        hostname.startsWith("172.") || 
        hostname.endsWith(".local");
        
    if (!isLocal && engineSelect) {
        engineSelect.value = "sparse";
        engineSelect.disabled = true;
        if (engineSelect.options.length > 0) {
            engineSelect.options[0].textContent = "Keyword (BM25) [Cloud Locked]";
        }
        engineSelect.style.cursor = "not-allowed";
        engineSelect.title = "Dense search is disabled on cloud deployment to prevent memory limit crashes.";
    }

    // Dynamic UI feedback for Conversational LLM switch status text
    const llmToggle = document.getElementById("llm-toggle");
    const llmStatusText = document.getElementById("llm-status-text");
    const llmRobotIcon = document.getElementById("llm-robot-icon");

    function updateLLMStatusUI() {
        if (!llmToggle || !llmStatusText) return;
        if (llmToggle.checked) {
            llmStatusText.textContent = "Conversational LLM: ON";
            llmStatusText.style.color = "#00e676";
            if (llmRobotIcon) llmRobotIcon.style.color = "#00e676";
        } else {
            llmStatusText.textContent = "Conversational LLM: OFF";
            llmStatusText.style.color = "";
            if (llmRobotIcon) llmRobotIcon.style.color = "";
        }
    }

    if (llmToggle) {
        llmToggle.addEventListener("change", updateLLMStatusUI);
        // Initial setup
        updateLLMStatusUI();
    }
    
    // ----------------------------------------------------
    // Timer helper functions
    // ----------------------------------------------------
    function startTimer() {
        secondsElapsed = 0;
        recordingTimer.textContent = "00:00";
        recordingTimer.classList.remove("hidden");
        
        timerInterval = setInterval(() => {
            secondsElapsed++;
            const mins = String(Math.floor(secondsElapsed / 60)).padStart(2, '0');
            const secs = String(secondsElapsed % 60).padStart(2, '0');
            recordingTimer.textContent = `${mins}:${secs}`;
        }, 1000);
    }
    
    function stopTimer() {
        if (timerInterval) {
            clearInterval(timerInterval);
            timerInterval = null;
        }
    }
    
    // ----------------------------------------------------
    // Speech Recognition & Voice Input Handler (Targeted Fix)
    // ----------------------------------------------------
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    let recognitionInstance = null;
    let activeStream = null;
    let finalTranscript = "";
    let interimTranscript = "";

    function getLanguageCode(selectedLang) {
        const LANG_MAP = {
            "en": "en-US", "hi": "hi-IN", "gu": "gu-IN", "ta": "ta-IN",
            "mr": "mr-IN", "ur": "ur-IN", "bn": "bn-IN", "kn": "kn-IN",
            "ml": "ml-IN", "pa": "pa-IN", "or": "or-IN", "as": "as-IN",
            "sa": "sa-IN", "ne": "ne-NP", "auto": "en-IN"
        };
        return LANG_MAP[selectedLang] || "en-US";
    }

    async function startRecording() {
        if (!SpeechRecognition) {
            console.warn("SpeechRecognition API is not supported in this browser.");
            if (recordingStatus) recordingStatus.textContent = "Speech recognition is not supported in this browser.";
            return;
        }

        // Clear previous input for a new recording session
        if (queryInput) queryInput.value = "";
        finalTranscript = "";
        interimTranscript = "";

        try {
            // Request mic access & stream
            activeStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            
            // Setup Speech Recognition instance
            recognitionInstance = new SpeechRecognition();
            recognitionInstance.continuous = true;
            recognitionInstance.interimResults = true;
            
            const langSelect = document.getElementById("lang-select");
            const selectedLang = langSelect ? langSelect.value : "auto";
            recognitionInstance.lang = getLanguageCode(selectedLang);

            recognitionInstance.onstart = () => {
                isRecording = true;
                if (micBtn) micBtn.classList.add("recording");
                if (recordingStatus) recordingStatus.textContent = "Listening... Speak your query";
                startTimer();
            };

            recognitionInstance.onresult = (event) => {
                interimTranscript = "";
                for (let i = event.resultIndex; i < event.results.length; ++i) {
                    const result = event.results[i];
                    if (result.isFinal) {
                        const txt = result[0].transcript.trim();
                        if (txt) finalTranscript += (finalTranscript ? " " : "") + txt;
                    } else {
                        const txt = result[0].transcript.trim();
                        if (txt) interimTranscript += (interimTranscript ? " " : "") + txt;
                    }
                }
                
                // Display live combined transcript inside existing input box
                if (queryInput) {
                    const fullText = finalTranscript + (interimTranscript ? (finalTranscript ? " " : "") + interimTranscript : "");
                    queryInput.value = fullText;
                }
                if (recordingStatus) recordingStatus.textContent = "Transcribing live speech...";
            };

            recognitionInstance.onerror = (event) => {
                console.warn("Speech recognition error:", event.error);
                let msg = "Speech recognition error.";
                if (event.error === "not-allowed") {
                    msg = "Microphone access is required for voice input.";
                } else if (event.error === "no-speech") {
                    msg = "No speech detected. Click mic to retry.";
                } else if (event.error === "audio-capture") {
                    msg = "No microphone found on device.";
                }
                if (recordingStatus) recordingStatus.textContent = msg;
            };

            recognitionInstance.onend = () => {
                if (isRecording) {
                    try {
                        recognitionInstance.start();
                    } catch (e) {
                        isRecording = false;
                        if (micBtn) micBtn.classList.remove("recording");
                        stopTimer();
                    }
                }
            };

            recognitionInstance.start();

        } catch (err) {
            console.error("Error accessing microphone:", err);
            if (recordingStatus) recordingStatus.textContent = "Microphone access is required for voice input.";
        }
    }

    function stopRecording() {
        isRecording = false;
        if (recognitionInstance) {
            try {
                recognitionInstance.stop();
            } catch (e) {}
            recognitionInstance = null;
        }

        if (activeStream) {
            activeStream.getTracks().forEach(track => track.stop());
            activeStream = null;
        }

        if (micBtn) micBtn.classList.remove("recording");
        stopTimer();

        if (queryInput) {
            const finalQuery = queryInput.value.trim();
            if (finalQuery) {
                if (recordingStatus) recordingStatus.textContent = "Recording finished. Click Analyze to process query.";
            } else {
                if (recordingStatus) recordingStatus.textContent = "No speech detected. Please try again.";
            }
        }
    }

    if (micBtn) {
        micBtn.addEventListener("click", () => {
            if (!isRecording) {
                startRecording();
            } else {
                stopRecording();
            }
        });
    }
    
    // ----------------------------------------------------
    // API Submit query functions
    // ----------------------------------------------------
    
    function showLoading(statusMsg) {
        loaderStatus.textContent = statusMsg;
        loader.classList.remove("hidden");
        answerDisplay.classList.add("hidden");
        
        // Reset latency and badges during load
        relevanceBadge.textContent = "Processing";
        relevanceBadge.className = "badge";
        latencyStt.textContent = "-";
        latencyRetrieval.textContent = "-";
        latencyGen.textContent = "-";
        latencyTotal.textContent = "-";
    }
    
    function hideLoading() {
        loader.classList.add("hidden");
        answerDisplay.classList.remove("hidden");
    }
    
    async function submitTextQuery(query) {
        showLoading("Searching vector index...");
        transcriptContainer.classList.add("hidden");
        
        const selectedLang = document.getElementById("lang-select").value;
        const useLLM = document.getElementById("llm-toggle").checked;
        const retrievalMode = engineSelect ? engineSelect.value : "default";
        
        try {
            const response = await fetch("/api/query", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ query: query, k: 5, language: selectedLang, use_llm: useLLM, retrieval_mode: retrievalMode })
            });
            
            if (!response.ok) {
                throw new Error(`Server returned error: ${response.status}`);
            }
            
            const data = await response.json();
            updateUI(data, false);
        } catch (err) {
            console.error("Error submitting text query:", err);
            renderError(err.message);
        } finally {
            hideLoading();
        }
    }
    
    async function submitVoiceQuery(audioBlob) {
        showLoading("Transcribing voice recording...");
        
        const selectedLang = document.getElementById("lang-select").value;
        const useLLM = document.getElementById("llm-toggle").checked;
        const retrievalMode = engineSelect ? engineSelect.value : "default";
        
        try {
            const formData = new FormData();
            // Provide filename so FastAPI can parse extension correctly
            formData.append("file", audioBlob, "query_recording.webm");
            formData.append("k", 5);
            formData.append("language", selectedLang);
            formData.append("use_llm", useLLM);
            formData.append("retrieval_mode", retrievalMode);
            
            const response = await fetch("/api/query-voice", {
                method: "POST",
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`Server returned error: ${response.status}`);
            }
            
            const data = await response.json();
            updateUI(data, true);
        } catch (err) {
            console.error("Error submitting voice query:", err);
            renderError(err.message);
        } finally {
            hideLoading();
            recordingStatus.textContent = "Click mic to start recording";
        }
    }
    
    // ----------------------------------------------------
    // UI Rendering functions
    // ----------------------------------------------------
    
    function updateUI(data, isVoice) {
        // 1. Render transcript if voice
        if (isVoice && data.transcript) {
            transcriptText.textContent = data.transcript;
            transcriptContainer.classList.remove("hidden");
        } else {
            transcriptContainer.classList.add("hidden");
        }

        // Render detected language badge if dynamic language mapping is active
        const selectedLang = document.getElementById("lang-select").value;
        if (selectedLang === "auto" && data.detected_language) {
            const langNames = {
                "en": "English", "hi": "Hindi", "gu": "Gujarati", "ta": "Tamil",
                "mr": "Marathi", "ur": "Urdu", "bn": "Bengali", "kn": "Kannada",
                "ml": "Malayalam", "pa": "Punjabi", "or": "Odia", "as": "Assamese",
                "sa": "Sanskrit", "ne": "Nepali"
            };
            const name = langNames[data.detected_language] || data.detected_language.toUpperCase();
            detectedLangBadge.textContent = `Detected: ${name}`;
            detectedLangBadge.classList.remove("hidden");
        } else {
            detectedLangBadge.classList.add("hidden");
        }
        
        // 2. Render answer
        answerDisplay.innerHTML = `<div class="answer-text">${data.answer}</div>`;
        
        // Update dynamic search engine label
        if (retrievalLabel && data.retrieval_mode) {
            if (data.retrieval_mode === "sparse") {
                retrievalLabel.innerHTML = `Lexical Retrieval <span class="badge" style="font-size:8px; padding:1px 4px; margin-left:4px; background:rgba(255,255,255,0.08); color:var(--text-secondary);">BM25</span>`;
            } else {
                retrievalLabel.innerHTML = `Vector Retrieval <span class="badge success" style="font-size:8px; padding:1px 4px; margin-left:4px; background:rgba(0,230,118,0.15); color:#00e676;">DENSE</span>`;
            }
        }

        // 3. Render relevance badge
        if (data.relevance_passed) {
            relevanceBadge.textContent = "Grounded";
            relevanceBadge.className = "badge success";
        } else {
            relevanceBadge.textContent = "Fallback Active";
            relevanceBadge.className = "badge error";
        }
        
        // 4. Render latency times
        if (isVoice && data.latency_ms.stt) {
            latencyStt.textContent = `${data.latency_ms.stt.toFixed(0)} ms`;
        } else {
            latencyStt.textContent = "N/A";
        }
        
        latencyRetrieval.textContent = `${data.latency_ms.retrieval.toFixed(0)} ms`;
        latencyGen.textContent = `${data.latency_ms.generation.toFixed(0)} ms`;
        latencyTotal.textContent = `${data.latency_ms.total_rag.toFixed(0)} ms`;
        
        // 5. Render sources
        sourcesCount.textContent = `${data.sources.length} Sources`;
        
        if (data.sources.length === 0) {
            sourcesList.innerHTML = `<div class="sources-placeholder">No sources retrieved.</div>`;
            return;
        }
        
        sourcesList.innerHTML = "";
        data.sources.forEach(src => {
            const scorePct = (src.score * 100).toFixed(0);
            const isSelected = src.metadata.is_selected === 1 || src.metadata.is_selected === "1";
            
            const sourceCard = document.createElement("div");
            sourceCard.className = "source-item";
            sourceCard.innerHTML = `
                <div class="source-meta">
                    <span class="source-id"><i class="fa-solid fa-file-lines"></i> ${src.document_id}</span>
                    <div class="source-score-container">
                        <span>Score: ${(src.score).toFixed(4)}</span>
                        <div class="score-meter" title="Match Score: ${scorePct}%">
                            <div class="score-fill" style="width: ${scorePct}%"></div>
                        </div>
                    </div>
                </div>
                <div class="source-text">${src.text}</div>
            `;
            sourcesList.appendChild(sourceCard);
        });
    }
    
    function renderError(errMsg) {
        answerDisplay.innerHTML = `
            <div class="answer-text" style="color: var(--status-error);">
                <i class="fa-solid fa-triangle-exclamation"></i> 
                <strong>Error processing request:</strong> ${errMsg}
            </div>
        `;
        relevanceBadge.textContent = "Error";
        relevanceBadge.className = "badge error";
    }
    
    // ----------------------------------------------------
    // Event listeners & Startup
    // ----------------------------------------------------
    queryForm.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = queryInput.value.trim();
        if (text) {
            submitTextQuery(text);
            queryInput.value = "";
        }
    });

    async function loadEvaluationStats() {
        try {
            const response = await fetch("/api/evaluation-results");
            if (!response.ok) {
                console.warn("Evaluation results not found on server.");
                return;
            }
            const data = await response.json();
            
            // Format timestamp nicely
            document.getElementById("eval-timestamp").textContent = `Tested: ${data.timestamp}`;
            
            // Retrieval Stats
            document.getElementById("eval-retrieval-p50").textContent = `${data.retrieval.p50_ms.toFixed(2)} ms`;
            document.getElementById("eval-retrieval-p70").textContent = `${data.retrieval.p70_ms.toFixed(2)} ms`;
            document.getElementById("eval-retrieval-p100").textContent = `${data.retrieval.p100_ms.toFixed(2)} ms`;
            document.getElementById("eval-recall").textContent = `${data.retrieval.recall_accuracy_percent.toFixed(2)}%`;
            
            // RAG Stats
            document.getElementById("eval-rag-p50").textContent = `${(data.rag.total.p50_ms / 1000).toFixed(2)} s`;
            document.getElementById("eval-rag-p70").textContent = `${(data.rag.total.p70_ms / 1000).toFixed(2)} s`;
            document.getElementById("eval-rag-p100").textContent = `${(data.rag.total.p100_ms / 1000).toFixed(2)} s`;
            document.getElementById("eval-count").textContent = `${data.rag.total_run} Queries`;
        } catch (err) {
            console.error("Error loading evaluation stats:", err);
        }
    }

    // ----------------------------------------------------
    // Dynamic Query Suggestion Chips
    // ----------------------------------------------------
    const langSelect = document.getElementById("lang-select");
    const suggestionChips = document.getElementById("suggestion-chips");

    const SUGGESTIONS = {
        "en": [
            "Why was the secret nuclear facility placed close to a large water body?",
            "what was the immediate impact of the success of the manhattan project?"
        ],
        "hi": [
            "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?",
            "विभिन्न प्रकार की सामाजिक सुरक्षा विकलांगता"
        ],
        "gu": [
            "મેનહટન પ્રોજેક્ટની સફળતાની તાત્કાલિક અસર શું હતી?",
            "વિવિધ પ્રકારની સામાજિક સુરક્ષા અક્ષમતા"
        ],
        "auto": [
            "Why was the secret nuclear facility placed close to a large water body?",
            "मैनहट्टन परियोजना की सफलता का तुरंत क्या प्रभाव पड़ा?",
            "મેનહટન પ્રોજેક્ટની સફળતાની તાત્કાલિક અસર શું હતી?"
        ]
    };

    function populateSuggestions() {
        if (!suggestionChips || !langSelect) return;
        suggestionChips.innerHTML = "";
        
        const selectedLang = langSelect.value;
        const queries = SUGGESTIONS[selectedLang] || SUGGESTIONS["auto"];
        
        queries.forEach(q => {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "suggestion-chip-btn";
            btn.innerHTML = `<i class="fa-solid fa-lightbulb"></i> <span>${q}</span>`;
            btn.addEventListener("click", () => {
                queryInput.value = q;
                submitTextQuery(q);
            });
            suggestionChips.appendChild(btn);
        });
    }

    if (langSelect) {
        langSelect.addEventListener("change", populateSuggestions);
        populateSuggestions(); // Populate on load
    }

    // Load overall benchmarks immediately on load
    loadEvaluationStats();
});
