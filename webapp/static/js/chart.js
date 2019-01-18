
function create_eeg(tag, element){
    $(function () {


        id = tag + element;
//        console.log(id)
        var step = Math.PI / 30,
        data = time().add(function(x) { return Math.cos(x) + 400; }),
        interval = null;

        var sizes = [
            { width: 400, height: 100 },
            { width: 800, height: 150 },
            { width: $(id + ' .epoch').width(), height: $(id + ' .epoch').height() }
        ];

        var chart = $(id + ' .epoch').epoch({
            type: 'time.line',
            data: data.get([0, 2*Math.PI], step),
            historySize: 30
        })

        function pushPoint() {
            console.log(step)
            console.log(data.next(step))
            chart.push(data.next(step));
        }

        $('#eeg-mean' + ' .playback').click(function(e) {
            if (!interval) {
                interval = setInterval(function() { pushPoint() }, 1000);
                pushPoint();
                $('#eeg-mean' + ' .playback').text('Pause');
            }
            else {
                clearInterval(interval);
                interval = null;
                $('#eeg-mean' + ' .playback').text('Play');
            }
        });

        $(id + ' .size').click(function(e) {
            var size = sizes[parseInt($(e.target).attr('data-index'))];
            chart.option('width', size.width);
            chart.option('height', size.height);
        });
    });
}


eeg_channel_list = ['Mean', 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'];
//channel_list = channel_list.map(v => v.toLowerCase())

eeg_channel_list.forEach((element) => {
    create_eeg('#eeg-', element.toLowerCase());
});



function create_analyzer(tag, element){
    $(function () {


        id = tag + element;
//        console.log(id)
        var step = Math.PI / 30,
        data2 = time().add(function(x) { return Math.cos(x) + 1; }),
        interval = null;

        var sizes = [
            { width: 400, height: 100 },
            { width: 800, height: 150 },
            { width: $(id + ' .epoch').width(), height: $(id + ' .epoch').height() }
        ];

//        var chart = $(id + ' .epoch').epoch({
//            type: 'time.line',
//            data: data.get([0, 2*Math.PI], step)
//        })
        var chart = $(id + ' .epoch').epoch({
            type: 'time.line',
            data: data().add(function(x) { return Math.sin(x); })
                        .add(function(x) { return Math.cos(x); })
                        .get([0, 2*Math.PI], Math.PI/25),
            axes: ['bottom', 'left']
        });

        console.log(data().add(function(x) { return Math.sin(x); })
                        .add(function(x) { return Math.cos(x); })
                        .get([0, 2*Math.PI], Math.PI/25))

        function pushPoint() {
            console.log(step)
            console.log(data2.next(step))
            chart.push(data2.next(step));
        }

        $('#eeg-mean' + ' .playback').click(function(e) {
            if (!interval) {
                interval = setInterval(function() { pushPoint() }, 1000);
                pushPoint();
                $('#eeg-mean' + ' .playback').text('Pause');
            }
            else {
                clearInterval(interval);
                interval = null;
                $('#eeg-mean' + ' .playback').text('Play');
            }
        });

        $(id + ' .size').click(function(e) {
            var size = sizes[parseInt($(e.target).attr('data-index'))];
            chart.option('width', size.width);
            chart.option('height', size.height);
        });
    });
}

analyzeChannelList = ['fun', 'difficulty', 'arousal', 'valence', 'immersion', 'emotion'];
analyzeChannelList.forEach((element) => {
    create_analyzer('#analyze-', element.toLowerCase())
});
