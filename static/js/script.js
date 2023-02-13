var lines = []
var clearBut
var predictBut
var loc = window.location.pathname
var dir = loc.substring(0, loc.lastIndexOf('/'))

function setup() {
    canvas = createCanvas(280, 280)
    strokeWeight(10)

    var options = createDiv().style("display: flex")
    clearBut = createButton("Clear image").parent(options)
    predictBut = createButton("Predict!").parent(options)
}

function draw() {
    background(0)

    clearBut.mousePressed(function () {
        lines = []
    })

    predictBut.mousePressed(function () {
        // get the current state of the canvas and convert it to blob

        var canvasData = canvas.elt.toDataURL()

        $(document).ready(function () {
            $.ajax({
                url: '',
                type: 'GET',
                contentType: 'application/json',
                data: {
                    canvasData: canvasData
                },
                success: function (response) {
                    console.log("GET request was successful.")
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

