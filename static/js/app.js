let demandChart;
let marketplaceChart;

async function checkHealth(){

    const response = await fetch("/health");

    const data = await response.json();

    const status = document.getElementById("api-status");

    if(data.model_loaded){

        status.innerHTML="● Connected";

        status.style.background="#22c55e";
    }
    else{

        status.innerHTML="● Offline";

        status.style.background="#ef4444";
    }
}

async function populateZones(){

    const response = await fetch(
        "/metadata/zones"
    );

    const data = await response.json();

    const select =
        document.getElementById("zone_id");

    select.innerHTML = data.zones
        .map(
            z =>
            `<option value="${z}">
                Zone ${z}
            </option>`
        )
        .join("");
}

function createCharts(){

    demandChart = new Chart(
        document.getElementById("demandChart"),
        {
            type:"line",
            data:{
                labels:["Demand"],
                datasets:[{
                    label:"Forecast Demand",
                    data:[0]
                }]
            }
        }
    );

    marketplaceChart = new Chart(
        document.getElementById("marketplaceChart"),
        {
            type:"bar",
            data:{
                labels:[
                    "Demand",
                    "Drivers"
                ],
                datasets:[{
                    label:"Marketplace",
                    data:[0,0]
                }]
            }
        }
    );
}

document
.getElementById("forecastBtn")
.addEventListener(
    "click",
    async ()=>{

        const payload = {

            input:{

                zone_id:Number(
                    document.getElementById(
                        "zone_id"
                    ).value
                ),

                timestamp:
                document.getElementById(
                    "timestamp"
                ).value
            }
        };

        const response =
            await fetch(
                "/forecast",
                {
                    method:"POST",
                    headers:{
                        "Content-Type":
                        "application/json"
                    },
                    body:JSON.stringify(
                        payload
                    )
                }
            );

        const result =
            await response.json();

        document.getElementById(
            "forecastDemand"
        ).innerHTML =
        result.forecast_demand.toFixed(0);

        document.getElementById(
            "availableDrivers"
        ).innerHTML =
        result.available_drivers.toFixed(0);

        document.getElementById(
            "driverGap"
        ).innerHTML =
        result.driver_gap.toFixed(0);

        document.getElementById(
            "recommendedSurge"
        ).innerHTML =
        result.recommended_surge.toFixed(2);

        document.getElementById(
            "waitTime"
        ).innerHTML =
        result.predicted_wait_time.toFixed(1);

        document.getElementById(
            "forecastRevenue"
        ).innerHTML =
        "$"+result.forecast_revenue.toFixed(0);

        document.getElementById(
            "marketplaceStatus"
        ).innerHTML =
        result.marketplace_status;

        document.getElementById(
            "supplyRatio"
        ).innerHTML =
        result.forecast_supply_ratio.toFixed(2);

        document.getElementById(
            "requiredDrivers"
        ).innerHTML =
        result.required_drivers;

        document.getElementById(
            "riskScore"
        ).innerHTML =
        result.risk_score.toFixed(2);

        document.getElementById(
            "shortagePct"
        ).innerHTML =
        (
            result.shortage_pct*100
        ).toFixed(0)+"%";

        document.getElementById(
            "zoneType"
        ).innerHTML =
        result.zone_type;

        demandChart.data.datasets[0]
        .data=[
            result.forecast_demand
        ];

        demandChart.update();

        marketplaceChart
        .data.datasets[0]
        .data=[
            result.forecast_demand,
            result.available_drivers
        ];

        marketplaceChart.update();
    }
);

async function init(){

    await checkHealth();

    await populateZones();

    createCharts();
}

init();