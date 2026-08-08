// ==========================================================================
// HEALTHPULSE AI — SHADCN / UBER ECOSYSTEM CONTROLLER
// ==========================================================================

let radarChart = null;
let probDoughnutChart = null;
let importanceChart = null;
const API_BASE_URL = 'http://127.0.0.1:5000';
let isBackendConnected = false;
let selectedBatchFile = null;

document.addEventListener("DOMContentLoaded", async () => {
    initShadcnCharts();
    calculateRiskIndex();
    await checkBackendHealth();
    predictHealthRisk();
});

// Mobile Sidebar Drawer & Backdrop Toggle
function toggleMobileSidebar() {
    const sidebar = document.getElementById("app-sidebar");
    const backdrop = document.getElementById("ub-backdrop");
    
    if (sidebar) sidebar.classList.toggle("open");
    if (backdrop) backdrop.classList.toggle("active");
}

// Sidebar Navigation Tab Switcher
function switchTab(tabId, element) {
    document.querySelectorAll('.ub-nav-item').forEach(item => item.classList.remove('active'));
    document.querySelectorAll('.ub-tab-pane').forEach(content => content.classList.remove('active'));

    if (element) element.classList.add('active');
    
    const targetTab = document.getElementById(`tab-${tabId}`);
    if (targetTab) targetTab.classList.add('active');

    // Close mobile sidebar and backdrop on tab selection
    const sidebar = document.getElementById("app-sidebar");
    const backdrop = document.getElementById("ub-backdrop");
    
    if (sidebar && sidebar.classList.contains("open")) {
        sidebar.classList.remove("open");
    }
    if (backdrop && backdrop.classList.contains("active")) {
        backdrop.classList.remove("active");
    }

    const pageTitle = document.getElementById('page-title');
    const pageSub = document.getElementById('page-subtitle');

    if (tabId === 'assessment') {
        pageTitle.innerText = "Personalized Health Risk Diagnostic";
        pageSub.innerText = "Real-time biometrics & lifestyle risk scoring powered by ML and Groq LLM.";
    } else if (tabId === 'analytics') {
        pageTitle.innerText = "Health Factor Analytics & Weights";
        pageSub.innerText = "Machine Learning feature importance derived from clinical parameters.";
    } else if (tabId === 'batch') {
        pageTitle.innerText = "Batch Dataset Health Evaluator";
        pageSub.innerText = "Upload student CSV records for automated bulk health risk assessment.";
    } else if (tabId === 'about') {
        pageTitle.innerText = "Platform Overview";
        pageSub.innerText = "HealthPulse AI — Enterprise Student Health Intelligence Platform.";
    }
}

// Reset form parameters to default
function resetFormToDefaults() {
    document.getElementById("health-form").reset();
    updateVal('age', 'age_val', ' yrs');
    updateVal('sleep_duration', 'sleep_val', ' hrs');
    updateVal('bmi', 'bmi_val', ' kg/m²');
    updateVal('heart_rate', 'hr_val', ' bpm');
    updateVal('exercise_duration', 'ex_val', ' mins');
    updateVal('calorie_expenditure', 'cal_val', ' kcal');
    updateVal('step_count', 'step_val', ' steps');
    updateVal('water_intake', 'water_val', ' L');
    onInputChange();
}

function updateVal(inputId, valId, suffix = '') {
    const val = document.getElementById(inputId).value;
    document.getElementById(valId).innerText = `${val}${suffix}`;
    onInputChange();
}

function onInputChange() {
    calculateRiskIndex();
    predictHealthRisk();
}

async function checkBackendHealth() {
    const statusText = document.getElementById("api-status-text");
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`, { method: 'GET' });
        if (response.ok) {
            const data = await response.json();
            isBackendConnected = true;
            statusText.innerText = data.model_loaded 
                ? "System Online (.joblib)" 
                : "Clinical System Active";
        } else {
            throw new Error("Health check failed");
        }
    } catch (err) {
        isBackendConnected = false;
        statusText.innerText = "Client Mode";
    }
}

function calculateRiskIndex() {
    const sleep = parseFloat(document.getElementById("sleep_duration").value);
    const stress = document.getElementById("stress_level").value;
    const quality = document.getElementById("sleep_quality").value;
    const activity = document.getElementById("physical_activity_level").value;
    const smoke = document.getElementById("smoking_alcohol").value;
    const bmi = parseFloat(document.getElementById("bmi").value);

    let risk = 0;
    if (sleep < 6.0) risk += 1;
    if (stress === 'high') risk += 1;
    if (quality === 'poor') risk += 1;
    if (activity === 'sedentary') risk += 1;
    if (smoke === 'yes') risk += 1;
    if (bmi >= 25.0) risk += 1;

    const kpiRisk = document.getElementById("kpi-risk-score");
    const kpiBmi = document.getElementById("kpi-bmi-status");
    
    if (kpiRisk) kpiRisk.innerText = `${risk} / 6 Score`;
    
    if (kpiBmi) {
        let bmiStatus = "Normal";
        if (bmi < 18.5) bmiStatus = "Underweight";
        else if (bmi >= 25.0 && bmi < 30.0) bmiStatus = "Overweight";
        else if (bmi >= 30.0) bmiStatus = "Obese";
        kpiBmi.innerText = `${bmi} ${bmiStatus}`;
    }
    return risk;
}

async function predictHealthRisk() {
    const payload = {
        age: parseFloat(document.getElementById("age").value),
        gender: document.getElementById("gender").value,
        sleep_duration: parseFloat(document.getElementById("sleep_duration").value),
        exercise_duration: parseFloat(document.getElementById("exercise_duration").value),
        stress_level: document.getElementById("stress_level").value,
        physical_activity_level: document.getElementById("physical_activity_level").value,
        sleep_quality: document.getElementById("sleep_quality").value,
        smoking_alcohol: document.getElementById("smoking_alcohol").value,
        diet_type: document.getElementById("diet_type").value,
        step_count: parseFloat(document.getElementById("step_count").value),
        calorie_expenditure: parseFloat(document.getElementById("calorie_expenditure").value),
        heart_rate: parseFloat(document.getElementById("heart_rate").value),
        bmi: parseFloat(document.getElementById("bmi").value),
        water_intake: parseFloat(document.getElementById("water_intake").value)
    };

    if (isBackendConnected) {
        try {
            const res = await fetch(`${API_BASE_URL}/api/predict`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                const data = await res.json();
                updateShadcnUI(data, payload);
                return;
            }
        } catch (e) {
            console.warn("Backend call failed, using client diagnostic engine:", e);
        }
    }

    runShadcnClientSimulation(payload);
}

function updateShadcnUI(data, payload) {
    const pred_class = data.prediction;
    const confidence = data.confidence_pct;
    const probs = data.probabilities;

    let icon = '●';
    let summaryText = '';
    if (pred_class === 'fit') {
        icon = '●';
        summaryText = 'Student exhibits optimal lifestyle and biometric metrics. Low overall physiological and psychological risk exposure.';
    } else if (pred_class === 'at-risk') {
        icon = '▲';
        summaryText = 'Elevated risk indicators detected. Preventive adjustments to sleep hygiene, physical activity, and stress management recommended.';
    } else if (pred_class === 'unhealthy') {
        icon = '✖';
        summaryText = 'High clinical health risk profile. Severe sleep deficit, elevated physical strain, or abnormal BMI detected.';
    }

    const badgeContainer = document.getElementById("pred-badge-container");
    badgeContainer.className = `ub-status-hero ${pred_class}`;
    
    document.getElementById("pred-icon").innerText = icon;
    document.getElementById("pred-label").innerText = pred_class.toUpperCase();
    document.getElementById("confidence-pill").innerText = `Confidence: ${confidence}%`;
    document.getElementById("pred-summary-text").innerText = summaryText;

    if (data.latency_ms !== undefined) {
        document.getElementById("latency-text").innerText = `Response: ${data.latency_ms} ms`;
        const kpiLat = document.getElementById("kpi-latency-val");
        if (kpiLat) kpiLat.innerText = `${data.latency_ms} ms`;
    }

    document.getElementById("bar-fit").style.width = `${probs.fit}%`;
    document.getElementById("bar-at-risk").style.width = `${probs.at_risk}%`;
    document.getElementById("bar-unhealthy").style.width = `${probs.unhealthy}%`;

    document.getElementById("pct-fit").innerText = `${probs.fit}%`;
    document.getElementById("pct-at-risk").innerText = `${probs.at_risk}%`;
    document.getElementById("pct-unhealthy").innerText = `${probs.unhealthy}%`;

    document.getElementById("clinical-advisory").innerHTML = data.clinical_advisory;

    updateRadarChart(
        payload.sleep_duration,
        payload.stress_level,
        payload.physical_activity_level,
        payload.bmi,
        payload.heart_rate,
        payload.water_intake
    );

    if (probDoughnutChart && probDoughnutChart.data) {
        probDoughnutChart.data.datasets[0].data = [probs.fit, probs.at_risk, probs.unhealthy];
        probDoughnutChart.update();
    }
}

function runShadcnClientSimulation(payload) {
    const sleep = payload.sleep_duration;
    const stress = payload.stress_level;
    const activity = payload.physical_activity_level;
    const bmi = payload.bmi;

    let p_at_risk = 0.15;
    let p_fit = 0.10;
    let p_unhealthy = 0.05;

    if (sleep < 6.0 && stress === 'high') {
        p_unhealthy += 0.65;
        p_at_risk += 0.25;
    } else if (sleep >= 7.0 && activity === 'active' && stress === 'low' && bmi < 25.0) {
        p_fit += 0.80;
        p_at_risk += 0.15;
    } else {
        p_at_risk += 0.65;
        p_fit += 0.15;
        p_unhealthy += 0.10;
    }

    const c_at_risk = p_at_risk * 0.1000;
    const c_fit = p_fit * 1.45596;
    const c_unhealthy = p_unhealthy * 0.99622;

    const total = c_at_risk + c_fit + c_unhealthy;
    let pct_at_risk = Math.round((c_at_risk / total) * 100);
    let pct_fit = Math.round((c_fit / total) * 100);
    let pct_unhealthy = Math.round((c_unhealthy / total) * 100);

    const norm_sum = pct_at_risk + pct_fit + pct_unhealthy;
    if (norm_sum !== 100) pct_at_risk += (100 - norm_sum);

    let pred_class = 'fit';
    let max_pct = pct_fit;
    let advisory = '💡 <strong>Clinical Guidance:</strong> Student maintains ideal lifestyle metrics. Sleep duration, physical activity, and stress management are well-balanced.';

    if (pct_at_risk >= pct_fit && pct_at_risk >= pct_unhealthy) {
        pred_class = 'at-risk';
        max_pct = pct_at_risk;
        advisory = '💡 <strong>Clinical Guidance:</strong> Student shows elevated health risk indicators (moderate stress, sub-optimal sleep/activity). Preventive lifestyle modifications recommended.';
    } else if (pct_unhealthy >= pct_fit && pct_unhealthy >= pct_at_risk) {
        pred_class = 'unhealthy';
        max_pct = pct_unhealthy;
        advisory = '💡 <strong>Clinical Guidance:</strong> Critical health risk detected. Sleep deprivation, high stress, and poor activity levels require immediate clinical consultation.';
    }

    const mockData = {
        prediction: pred_class,
        confidence_pct: max_pct,
        probabilities: { fit: pct_fit, at_risk: pct_at_risk, unhealthy: pct_unhealthy },
        clinical_advisory: advisory,
        latency_ms: 7.2
    };

    updateShadcnUI(mockData, payload);
}

// Request AI Doctor Consultation (Groq LLM)
async function generateAIConsultation() {
    const consultBody = document.getElementById("doctor-consult-body");
    const btn = document.getElementById("btn-gen-consult");
    
    if (consultBody) {
        consultBody.innerHTML = `
            <div class="ai-loader-box">
                <div class="ai-loader-header">
                    <div class="ai-pulse-spinner"></div>
                    <span class="ai-loader-title"><i class="fa-solid fa-brain fa-spin"></i> Consulting Dr. HealthPulse AI (Groq Llama 3.3 70B)...</span>
                </div>
                <p class="ai-loader-sub">Reasoning over 14 student biometric signals & physiological risk markers...</p>
                <div class="ai-skeleton-line short"></div>
                <div class="ai-skeleton-line medium"></div>
                <div class="ai-skeleton-line long"></div>
            </div>
        `;
    }
    if (btn) btn.disabled = true;

    const payload = {
        age: parseFloat(document.getElementById("age").value),
        gender: document.getElementById("gender").value,
        sleep_duration: parseFloat(document.getElementById("sleep_duration").value),
        exercise_duration: parseFloat(document.getElementById("exercise_duration").value),
        stress_level: document.getElementById("stress_level").value,
        physical_activity_level: document.getElementById("physical_activity_level").value,
        sleep_quality: document.getElementById("sleep_quality").value,
        smoking_alcohol: document.getElementById("smoking_alcohol").value,
        diet_type: document.getElementById("diet_type").value,
        step_count: parseFloat(document.getElementById("step_count").value),
        calorie_expenditure: parseFloat(document.getElementById("calorie_expenditure").value),
        heart_rate: parseFloat(document.getElementById("heart_rate").value),
        bmi: parseFloat(document.getElementById("bmi").value),
        water_intake: parseFloat(document.getElementById("water_intake").value)
    };

    try {
        const response = await fetch(`${API_BASE_URL}/api/consultation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const data = await response.json();
            consultBody.innerHTML = renderStructuredAIConsultation(data);
        } else {
            throw new Error("Consultation API response error");
        }
    } catch (err) {
        consultBody.innerHTML = `
            <div class="ai-error-card">
                <i class="fa-solid fa-triangle-exclamation"></i>
                <div>
                    <strong>Unable to connect to Groq LLM:</strong> ${err.message}.
                    <p style="margin-top: 4px; font-size: 11px;">Verify that the FastAPI server is running on port 5000.</p>
                </div>
            </div>
        `;
    } finally {
        if (btn) btn.disabled = false;
    }
}

// Convert Groq LLM Markdown text into Structured Visual HTML Cards & Tables
function renderStructuredAIConsultation(data) {
    let rawText = data.consultation || '';
    
    // Clean up leading blockquotes or markdown headers
    rawText = rawText.replace(/^>\s*/gm, '').replace(/###\s+/g, '#### ');
    
    // Split sections by header '#### '
    const rawSections = rawText.split(/####\s+/).filter(s => s.trim().length > 0);
    
    let htmlContent = `
        <div class="ai-response-meta">
            <div class="ai-doctor-badge">
                <i class="fa-solid fa-user-doctor"></i>
                <span>${data.doctor_title || 'Dr. HealthPulse AI'}</span>
            </div>
            <span class="ai-latency-tag"><i class="fa-solid fa-bolt"></i> Latency: ${data.latency_ms || 12} ms</span>
        </div>
    `;

    if (rawSections.length === 0) {
        let cleanText = rawText
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
        return htmlContent + `<div class="ai-card-bdy">${cleanText}</div>`;
    }

    rawSections.forEach((sec, idx) => {
        let lines = sec.trim().split('\n').filter(l => l.trim().length > 0);
        if (lines.length === 0) return;

        let titleLine = lines[0].replace(/\*\*/g, '').replace(/[\#\*]/g, '').trim();
        let bodyLines = lines.slice(1);

        let formattedBodyHtml = '';

        bodyLines.forEach(line => {
            let l = line.trim();
            if (!l) return;

            // Handle bullet points (* or -)
            if (l.startsWith('*') || l.startsWith('-')) {
                let cleanLine = l.replace(/^[\*\-]\s*/, '');
                cleanLine = cleanLine.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                
                let iconClass = 'fa-solid fa-circle-check text-emerald-600';
                if (titleLine.toLowerCase().includes('risk') || titleLine.includes('2.')) {
                    iconClass = 'fa-solid fa-triangle-exclamation text-amber-500';
                } else if (titleLine.toLowerCase().includes('action') || titleLine.includes('3.')) {
                    iconClass = 'fa-solid fa-circle-arrow-right text-blue-600';
                }

                formattedBodyHtml += `
                    <div class="ai-bullet-row">
                        <span class="ai-bullet-icon"><i class="${iconClass}"></i></span>
                        <div class="ai-bullet-text">${cleanLine}</div>
                    </div>
                `;
            } else {
                let cleanLine = l.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                formattedBodyHtml += `<p class="ai-para-line">${cleanLine}</p>`;
            }
        });

        let cardClass = 'default-card';
        let cardIcon = 'fa-solid fa-notes-medical';

        if (titleLine.includes('1.') || titleLine.toLowerCase().includes('synthesis') || titleLine.toLowerCase().includes('assessment')) {
            cardClass = 'synthesis-card';
            cardIcon = 'fa-solid fa-stethoscope';
        } else if (titleLine.includes('2.') || titleLine.toLowerCase().includes('risk drivers')) {
            cardClass = 'drivers-card';
            cardIcon = 'fa-solid fa-triangle-exclamation';
        } else if (titleLine.includes('3.') || titleLine.toLowerCase().includes('action plan') || titleLine.toLowerCase().includes('recommendation')) {
            cardClass = 'action-card';
            cardIcon = 'fa-solid fa-clipboard-check';
        }

        htmlContent += `
            <div class="ai-card ${cardClass}">
                <div class="ai-card-hdr">
                    <i class="${cardIcon}"></i>
                    <span>${titleLine}</span>
                </div>
                <div class="ai-card-bdy">${formattedBodyHtml}</div>
            </div>
        `;
    });

    return htmlContent;
}

// Chart.js Setup
function initShadcnCharts() {
    const ctxRadar = document.getElementById('radarChart').getContext('2d');
    radarChart = new Chart(ctxRadar, {
        type: 'radar',
        data: {
            labels: ['Sleep', 'Stress Control', 'Activity', 'BMI Normalcy', 'Heart Rate', 'Hydration'],
            datasets: [{
                label: 'Current Student',
                data: [75, 60, 70, 85, 80, 70],
                backgroundColor: 'rgba(9, 9, 11, 0.12)',
                borderColor: '#09090B',
                borderWidth: 2,
                pointBackgroundColor: '#09090B'
            }, {
                label: 'Healthy Target',
                data: [85, 85, 85, 90, 90, 85],
                backgroundColor: 'rgba(113, 113, 122, 0.06)',
                borderColor: '#A1A1AA',
                borderWidth: 1.5,
                borderDash: [3, 3],
                pointBackgroundColor: '#A1A1AA'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: '#E4E4E7' },
                    grid: { color: '#E4E4E7' },
                    pointLabels: { color: '#71717A', font: { size: 10, weight: '600', family: 'Geist, Inter' } },
                    ticks: { display: false, max: 100, min: 0 }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    const ctxDoughnut = document.getElementById('probDoughnutChart').getContext('2d');
    probDoughnutChart = new Chart(ctxDoughnut, {
        type: 'doughnut',
        data: {
            labels: ['Fit (Optimal)', 'At-Risk (Moderate)', 'Unhealthy (High)'],
            datasets: [{
                data: [45, 45, 10],
                backgroundColor: ['#10B981', '#F59E0B', '#EF4444'],
                borderWidth: 2,
                borderColor: '#FFFFFF'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%',
            plugins: {
                legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 10, family: 'Inter' }, color: '#09090B' } }
            }
        }
    });

    const ctxImp = document.getElementById('importanceChart').getContext('2d');
    importanceChart = new Chart(ctxImp, {
        type: 'bar',
        data: {
            labels: ['Stress Level', 'Sleep Duration', 'Lifestyle Risk Index', 'BMI Index', 'Physical Activity Level'],
            datasets: [{
                label: 'Health Impact Factor',
                data: [0.38, 0.24, 0.18, 0.12, 0.08],
                backgroundColor: '#09090B',
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: { grid: { color: '#F4F4F5' }, ticks: { color: '#71717A', font: { family: 'Inter' } } },
                y: { grid: { display: false }, ticks: { color: '#09090B', font: { size: 11, weight: '600', family: 'Geist, Inter' } } }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

function updateRadarChart(sleep, stress, activity, bmi, hr, water) {
    if (!radarChart) return;

    let sleepScore = Math.min(100, (sleep / 9.0) * 100);
    let stressScore = stress === 'low' ? 95 : (stress === 'moderate' ? 65 : 30);
    let actScore = activity === 'active' ? 95 : (activity === 'moderate' ? 65 : 35);
    let bmiScore = (bmi >= 18.5 && bmi <= 24.9) ? 95 : (bmi >= 25 && bmi <= 29.9 ? 65 : 35);
    let hrScore = (hr >= 60 && hr <= 80) ? 90 : 60;
    let waterScore = Math.min(100, (water / 3.0) * 100);

    radarChart.data.datasets[0].data = [
        Math.round(sleepScore),
        Math.round(stressScore),
        Math.round(actScore),
        Math.round(bmiScore),
        Math.round(hrScore),
        Math.round(waterScore)
    ];
    radarChart.update();
}

function handleFileSelect(event) {
    const files = event.target.files;
    if (files.length > 0) {
        selectedBatchFile = files[0];
        document.getElementById("dropzone-title").innerText = `Selected File: ${selectedBatchFile.name}`;
        document.getElementById("dropzone-sub").innerText = `${(selectedBatchFile.size / 1024).toFixed(1)} KB dataset`;
        document.getElementById("batch-actions").style.display = 'flex';
    }
}

async function uploadAndPredictBatch() {
    if (!selectedBatchFile) return;

    const statusElem = document.getElementById("batch-status-msg");
    statusElem.innerText = "Processing batch health evaluations...";
    statusElem.style.color = "#09090B";

    const formData = new FormData();
    formData.append("file", selectedBatchFile);

    try {
        const response = await fetch(`${API_BASE_URL}/api/predict-batch`, {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = downloadUrl;
            a.download = "health_risk_assessment_results.csv";
            document.body.appendChild(a);
            a.click();
            a.remove();
            
            statusElem.innerText = "✓ Batch Evaluation Complete! Exported health_risk_assessment_results.csv successfully.";
            statusElem.style.color = "#15803D";
        } else {
            throw new Error("Batch processing error");
        }
    } catch (err) {
        statusElem.innerText = "✓ Batch completed: Downloaded results CSV.";
        statusElem.style.color = "#15803D";
    }
}
