var lines = []
var clearBut
var predictBut
var loc = window.location.pathname
var dir = loc.substring(0, loc.lastIndexOf('/'))

function setup() {
    canvas = createCanvas(200, 200)
    strokeWeight(13)

    var options = createDiv().style("display: flex")
    predictBut = createButton("Predict").parent(options)
    clearBut = createButton("Clear Canvas").parent(options)
}

function draw() {
    background(0)

    clearBut.mousePressed(function () {
        lines = []
    })

    predictBut.mousePressed(function () {

        var canvasData = canvas.elt.toDataURL()

        $(document).ready(function () {
            $.ajax({
                url: '/predict',
                type: 'GET',
                contentType: 'application/json',
                data: {
                    canvasData: canvasData
                },
                success: function (response) {
                    console.log(response)
                    $('#result_digit').text('Model predicted: ' + response.pred_digit);
                    $('#result_prob').text('With probability: ' + Math.round(1000*response.probs[0][int(response.pred_digit)])/10 + "%");
                },
                error: function (xhr, status, error) {
                    console.log("https://i.kym-cdn.com/entries/icons/original/000/008/342/ihave.jpg")
                }
            })
        })
    })


    if (mouseIsPressed) {
        var line = new MyLine
        lines.push(line)
    }

    for (var line of lines) {
        line.show()
    }
}
