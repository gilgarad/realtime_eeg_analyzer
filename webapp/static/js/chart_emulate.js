$(document).ready(function(){
    var socket = io.connect("http://" + document.domain + ":" + location.port + "/update_data");
    var channel_list = ['Mean', 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'];
    var analyzeChannelList = ['fun', 'difficulty', 'arousal', 'valence', 'immersion', 'emotion'];

    eeg_chart_list = [];
    analyzerChartList = [];

    channel_list.forEach((element) => {
        //console.log('element: ' + element);
        id = '#eeg-' + element.toLowerCase();
        var chart = $(id + ' .epoch').epoch({
            type: 'time.line',
            data: [{label: 'eeg-' + element, values:[]}],
            axis: ['bottomm', 'left']
        });

        eeg_chart_list.push(chart);
    });

    analyzeChannelList.forEach((element) => {
        console.log('element: ' + element);
        id = '#analyze-' + element.toLowerCase()
        var chart = $(id + ' .epoch').epoch({
            type: 'time.line',
            data: [{label: 'a-' + element, values:[]},
                    {label: 'b-' + element, values:[]}],
            axes: ['bottom', 'left']
        });

        analyzerChartList.push(chart);
    });

    socket.on('response', function(msg) {
        //console.log('Received message');
        console.log('chart_list.length:' + chart_list.length);
        //console.log('channel_list.length:' + channel_list.length);
        var curTime = parseInt(new Date().getTime() / 1000)
        eeg_chart_list.forEach((chart, index) => {

            if (index == 0) {
                var newData = [{time: curTime, y: msg.eeg_mean}];
                chart.push(newData);
            } else {
                var newData = [{time: curTime, y: msg.eeg_channels[index - 1]}];
                chart.push(newData);
            }
        });

        console.log('analyzerChartList.length:' + analyzerChartList.length);
        analyzerChartList.forEach((chart, index) => {
            var newData = [{time: curTime, y: msg.eeg_mean},
                            {time: curTime, y: msg.eeg_channels[0]}];
            chart.push(newData);

        });
    });
});