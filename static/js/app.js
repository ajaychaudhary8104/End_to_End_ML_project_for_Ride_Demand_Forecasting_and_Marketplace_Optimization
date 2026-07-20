// ==========================================================
// CONFIG
// ==========================================================

const API_BASE = "";

// ==========================================================
// DOM ELEMENTS
// ==========================================================

const zoneSelect = document.getElementById("zone_id");
const timestampInput = document.getElementById("timestamp");
const forecastBtn = document.getElementById("forecastBtn");
const apiStatus = document.getElementById("api-status");

// ==========================================================
// CHARTS
// ==========================================================

let demandChart = null;
let marketplaceChart = null;

// ==========================================================
// INITIALIZATION
// ==========================================================

document.addEventListener("DOMContentLoaded", async () => {

    initializeTimestamp();

    await loadZones();

    await checkHealth();

    initializeCharts();

    forecastBtn.addEventListener(
        "click",
        generateForecast
    );
});

// ==========================================================
// TIMESTAMP
// ==========================================================

function initializeTimestamp() {

    const now = new Date();

    now.setMinutes(
        now.getMinutes() -
        now.getTimezoneOffset()
    );

    timestampInput.value =
        now.toISOString().slice(0, 16);
}

// ==========================================================
// LOAD ZONES
// ==========================================================

async function loadZones() {

    try {

        const response = await fetch(
            "/metadata/zones"
        );

        const data =
            await response.json();

        zoneSelect.innerHTML = "";

        data.zones.forEach(zone => {

            const option =
                document.createElement("option");

            option.value = zone;

            option.textContent =
                `Zone ${zone}`;

            zoneSelect.appendChild(option);
        });

    } catch (error) {

        console.error(
            "Failed loading zones:",
            error
        );

        for (let i = 1; i <= 263; i++) {

            const option =
                document.createElement("option");

            option.value = i;

            option.textContent =
                `Zone ${i}`;

            zoneSelect.appendChild(option);
        }
    }
}

// ==========================================================
// HEALTH CHECK
// ==========================================================

async function checkHealth() {

    try {

        const response =
            await fetch("/health");

        if (!response.ok) {
            throw new Error();
        }

        const data =
            await response.json();

        apiStatus.textContent =
            "API Online";

        apiStatus.className =
            "status online";

        console.log(
            "Health:",
            data
        );

    } catch {

        apiStatus.textContent =
            "API Offline";

        apiStatus.className =
            "status offline";
    }
}

// ==========================================================
// CHARTS
// ==========================================================

function initializeCharts() {

    const demandCtx =
        document
        .getElementById("demandChart")
        .getContext("2d");

    demandChart = new Chart(
        demandCtx,
        {
            type: "bar",

            data: {

                labels: [
                    "Forecast Demand",
                    "Available Drivers"
                ],

                datasets: [
                    {
                        label: "Marketplace",

                        data: [0, 0]
                    }
                ]
            },

            options: {

                responsive: true,

                maintainAspectRatio: false
            }
        }
    );

    const marketCtx =
        document
        .getElementById("marketplaceChart")
        .getContext("2d");

    marketplaceChart =
        new Chart(
            marketCtx,
            {
                type: "doughnut",

                data: {

                    labels: [
                        "Supply Ratio",
                        "Risk Score"
                    ],

                    datasets: [
                        {
                            data: [1, 0]
                        }
                    ]
                },

                options: {

                    responsive: true,

                    maintainAspectRatio: false
                }
            }
        );
}

// ==========================================================
// FORECAST
// ==========================================================

async function generateForecast() {

    try {

        // ----------------------------------
        // Validation
        // ----------------------------------

        if (!zoneSelect.value) {

            alert(
                "Please select a zone."
            );

            return;
        }

        if (!timestampInput.value) {

            alert(
                "Please select a timestamp."
            );

            return;
        }

        forecastBtn.disabled = true;

        forecastBtn.innerText =
            "Generating...";

        // ----------------------------------
        // Payload
        // ----------------------------------

        const payload = {

            zone_id:
                Number(
                    zoneSelect.value
                ),

            timestamp:
                timestampInput.value
        };

        console.log(
            "Sending Payload:",
            payload
        );

        // ----------------------------------
        // API Call
        // ----------------------------------

        const response =
            await fetch(
                "/forecast",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                        "application/json"
                    },

                    body:
                    JSON.stringify(
                        payload
                    )
                }
            );

        const data =
            await response.json();

        console.log(
            "Response:",
            data
        );

        if (!response.ok) {

            throw new Error(
                JSON.stringify(data)
            );
        }

        updateDashboard(data);

    } catch (error) {

        console.error(error);

        alert(
            "Forecast failed.\n\nCheck browser console."
        );

    } finally {

        forecastBtn.disabled = false;

        forecastBtn.innerText =
            "Generate Forecast";
    }
}

// ==========================================================
// DASHBOARD
// ==========================================================

function updateDashboard(data) {

    setValue(
        "forecastDemand",
        formatNumber(
            data.forecast_demand
        )
    );

    setValue(
        "availableDrivers",
        formatNumber(
            data.available_drivers
        )
    );

    setValue(
        "driverGap",
        formatNumber(
            data.driver_gap
        )
    );

    setValue(
        "recommendedSurge",
        `${data.recommended_surge}x`
    );

    setValue(
        "waitTime",
        `${Number(
            data.predicted_wait_time
        ).toFixed(1)} min`
    );

    setValue(
        "forecastRevenue",
        formatCurrency(
            data.forecast_revenue
        )
    );

    setValue(
        "marketplaceStatus",
        data.marketplace_status
    );

    setValue(
        "supplyRatio",
        Number(
            data.forecast_supply_ratio
        ).toFixed(2)
    );

    setValue(
        "requiredDrivers",
        data.required_drivers
    );

    setValue(
        "riskScore",
        `${(
            data.risk_score * 100
        ).toFixed(1)}%`
    );

    setValue(
        "shortagePct",
        `${(
            data.shortage_pct * 100
        ).toFixed(1)}%`
    );

    setValue(
        "zoneType",
        data.risk_score > 0.7
            ? "High Risk Zone"
            : "Normal Zone"
    );

    updateCharts(data);
}

// ==========================================================
// CHART UPDATE
// ==========================================================

function updateCharts(data) {

    demandChart.data.datasets[0].data = [

        Number(
            data.forecast_demand
        ),

        Number(
            data.available_drivers
        )
    ];

    demandChart.update();

    marketplaceChart.data.datasets[0].data = [

        Number(
            data.forecast_supply_ratio
        ),

        Number(
            data.risk_score
        )
    ];

    marketplaceChart.update();
}

// ==========================================================
// HELPERS
// ==========================================================

function setValue(id, value) {

    const element =
        document.getElementById(id);

    if (element) {
        element.textContent = value;
    }
}

function formatNumber(value) {

    return Number(
        value || 0
    ).toLocaleString();
}

function formatCurrency(value) {

    return "$" +
        Number(
            value || 0
        ).toLocaleString(
            undefined,
            {
                maximumFractionDigits: 2
            }
        );
}