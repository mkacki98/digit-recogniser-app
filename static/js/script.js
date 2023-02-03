var lines = []
var clearBut
var saveBut
var loc = window.location.pathname
var dir = loc.substring(0, loc.lastIndexOf('/'))

function setup() {
    createCanvas(280, 280)
    strokeWeight(10)

    var options = createDiv().style("display: flex")
    clearBut = createButton("Clear image").parent(options)
    saveBut = createButton("Save image").parent(options)
}

function draw() {
    background(0)

    clearBut.mousePressed(function () {
        lines = []
    })

    saveBut.mousePressed(function () {
        saveCanvas("drawing", "jpg")
    })

    if (mouseIsPressed) {
        var line = new MyLine
        lines.push(line)
    }

    for (var line of lines) {
        line.show()
    }
}