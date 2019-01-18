var myChart = new Chart(document.getElementById("myChart"), {
    type: 'bar',
    data: {
        labels: ["Red", "Blue", "Yellow", "Green", "Purple", "Orange"],
        datasets: [{
            label: '# of Votes',
            data: [12, 19, 3, 5, 2, 3],
            backgroundColor: [
                'rgba(255, 99, 132, 0.2)',
                'rgba(54, 162, 235, 0.2)',
                'rgba(255, 206, 86, 0.2)',
                'rgba(75, 192, 192, 0.2)',
                'rgba(153, 102, 255, 0.2)',
                'rgba(255, 159, 64, 0.2)'
            ],
            borderColor: [
                'rgba(255,99,132,1)',
                'rgba(54, 162, 235, 1)',
                'rgba(255, 206, 86, 1)',
                'rgba(75, 192, 192, 1)',
                'rgba(153, 102, 255, 1)',
                'rgba(255, 159, 64, 1)'
            ],
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            yAxes: [{
                ticks: {
                    beginAtZero:true
                }
            }]
        }
    }
});

//bar
var myBarChart = new Chart(document.getElementById("barChart").getContext('2d'), {
    type: 'bar',
    data: {
      labels: ["Happy", "Neutral", "Annoyed"],
      datasets: [{
        label: '게임 결과 Count',
        data: [31, 104, 57],
        backgroundColor: [
          'rgba(255, 99, 132, 0.2)',
          'rgba(54, 162, 235, 0.2)',
          'rgba(255, 206, 86, 0.2)',
          'rgba(75, 192, 192, 0.2)',
          'rgba(153, 102, 255, 0.2)',
          'rgba(255, 159, 64, 0.2)'
        ],
        borderColor: [
          'rgba(255,99,132,1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
          'rgba(255, 159, 64, 1)'
        ],
        borderWidth: 1
      }]
    },
    options: {
      scales: {
        yAxes: [{
          ticks: {
            beginAtZero: true
          }
        }]
      }
    }
});


//polar
var myPolarChart = new Chart(document.getElementById("polarChart").getContext('2d'), {
    type: 'polarArea',
    data: {
      labels: ["Red", "Green", "Yellow", "Grey", "Dark Grey"],
      datasets: [{
        data: [300, 50, 100, 40, 120],
        backgroundColor: ["rgba(219, 0, 0, 0.1)", "rgba(0, 165, 2, 0.1)", "rgba(255, 195, 15, 0.2)",
          "rgba(55, 59, 66, 0.1)", "rgba(0, 0, 0, 0.3)"
        ],
        hoverBackgroundColor: ["rgba(219, 0, 0, 0.2)", "rgba(0, 165, 2, 0.2)",
          "rgba(255, 195, 15, 0.3)", "rgba(55, 59, 66, 0.1)", "rgba(0, 0, 0, 0.4)"
        ]
      }]
    },
    options: {
      responsive: true
    }
});

//line
var labels = Array.from({length: 180}, (x, i) => i)
var rand_num = Array.from({length: 180}, (x, i) => (a = Math.random()) > 0.8 ? a - 0.8 : 0)

var myLineChart = new Chart(document.getElementById("lineChart").getContext('2d'), {
    type: 'line',
        data: {
          labels: labels,
          datasets: [{
            label: 'Overall 재미있는 구간',
            data: rand_num,
              label3: "누적 회수 그래프",
              data3: [0, 0, 1, 2, 3, 4, 5, 4, 3, 4, 3, 2, 1, 2, 1, 2, 1, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3],
              label2: "확률 as 재미있는 정도",
              data2: [0.67, 0.52, 0.88, 0.96, 0.94, 0.91, 0.56, 0.84, 0.23, 0.36, 0.49, 0.41, 0.87, 0.73, 0.54, 0.67, 0.52, 0.88, 0.13, 0.22, 0.54, 0.56, 0.37, 0.07, 0.24, 0.49, 0.41, 0.87, 0.73, 0.54],
              backgroundColor: [
                'rgba(105, 0, 132, .2)',
              ],
              borderColor: [
                'rgba(200, 99, 132, .7)',
              ],
              borderWidth: 2
            }
          ]
    },
    options: {
      responsive: true
    }
});

window.onload = function () {

        var dps = []; // dataPoints
        var chart = new CanvasJS.Chart("chartContainer", {
            title :{
                text: "Dynamic Data"
            },
            axisY: {
                includeZero: false
            },
            data: [{
                type: "line",
                dataPoints: dps
            }]
        });

        var xVal = 0;
        var yVal = 100;
        var updateInterval = 1000;
        var dataLength = 20; // number of dataPoints visible at any point

        var updateChart = function (count) {

            count = count || 1;


            for (var j = 0; j < count; j++) {
                yVal = yVal +  Math.round(5 + Math.random() *(-5-5));
                dps.push({
                    x: xVal,
                    y: yVal
                });
                xVal++;
            }


            if (dps.length > dataLength) {
                dps.shift();
            }

            chart.render();
        };

        updateChart(dataLength);
        setInterval(function(){updateChart()}, updateInterval);

        }


//var myChart = $('#myLineChart').epoch({ type: 'time.line', data: myData });
//
////{time:1370044800, y:100}
//t = 1370044800
//y = 100
//var new_data( () => {
//    t = t + 1
//    y = y + 1
//    return {time: t, y:y}
//})
//
//$('#toggle').click(function (e) {
//            // This switches the class names...
//            var myChart = $('#lineChart').epoch({ type: 'time.line', data: myData });
//
//            myChart.push(new_data())
//
//            // And this is required to see the updated styles...
//            myChart.redraw();
//          });


//pie
var myPieChart = new Chart(document.getElementById("pieChart").getContext('2d'), {
    type: 'pie',
    data: {
      labels: ["재미있음", "보통"],
      datasets: [{
        data: [54, 129],
        backgroundColor: ["#F7464A", "#46BFBD", "#FDB45C", "#949FB1", "#4D5360"],
        hoverBackgroundColor: ["#FF5A5E", "#5AD3D1", "#FFC870", "#A8B3C5", "#616774"]
      }]
    },
    options: {
      responsive: true
    }
});

var myHorizontalChart = new Chart(document.getElementById("horizontalBar"), {
        type: "horizontalBar",
        data: {
          labels: ["제 점수는요...?"],
          datasets: [{
            label: "분석 결과",
            data: [74],
            fill: false,
            backgroundColor: ["rgba(255, 159, 64, 0.2)"
            ],
            borderColor: ["rgb(255, 99, 132)", "rgb(255, 159, 64)", "rgb(255, 205, 86)",
              "rgb(75, 192, 192)", "rgb(54, 162, 235)", "rgb(153, 102, 255)", "rgb(201, 203, 207)"
            ],
            borderWidth: 1
          }]
        },
        options: {
          scales: {
            xAxes: [{
              ticks: {
                beginAtZero: true,
                max: 100
              }
            }]
          }
        }
      });