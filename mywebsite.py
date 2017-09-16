import os

def greedy_script(nickname,k_type,splitid,quantile,perfect,even,fname):
    return '''
<!DOCTYPE html>
<html>
<meta charset="utf-8">

<style>
body {{
  font: 11px sans-serif;
}}

.axis path,
.axis line {{
  fill: none;
  stroke: #000;
  shape-rendering: crispEdges;
}}

.dot {{
  stroke: #000;
}}

.tooltip {{
  position: absolute;
  width: 200px;
  height: 28px;
  pointer-events: none;
}}
</style>
<body>
<script src="http://d3js.org/d3.v3.min.js"></script>

<script>
var margin = {{top: 20, right: 20, bottom: 30, left: 40}},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

/* 
 * value accessor - returns the value to encode for a given data object.
 * scale - maps value to a visual display encoding, such as a pixel position.
 * map function - maps from data value to display value
 * axis - sets up axis
 */ 

// setup x 
var xValue = function(d) {{ return d.tsnex;}}, // data -> value
    xScale = d3.scale.linear().range([0, width]), // value -> display
    xMap = function(d) {{ return xScale(xValue(d));}}, // data -> display
    xAxis = d3.svg.axis().scale(xScale).orient("bottom");

// setup y
var yValue = function(d) {{ return d.tsney;}}, // data -> value
    yScale = d3.scale.linear().range([height, 0]), // value -> display
    yMap = function(d) {{ return yScale(yValue(d));}}, // data -> display
    yAxis = d3.svg.axis().scale(yScale).orient("left");


// setup fill color
var cValue = function(d) {{ return d.cluster;}},
    color = d3.scale.category10();

// add the graph canvas to the body of the webpage
var svg = d3.select("body").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// add the tooltip area to the webpage
var tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

// load data
//
d3.csv("{}", function(error, data) {{

  // change string (from CSV) into number format
  data.forEach(function(d) {{
    d.tsnex = +d.tsnex;
    d.tsney = +d.tsney;
    d.cluster = +d.cluster;
    d.clust_purity = +d.clust_purity;
    d.cat_purity = +d.cat_purity;
    d.correctness = +d.correctness;
    d.label = d.label;
    d.url = d.url;
    //console.log(d.url);
  }});

  // don't want dots overlapping axis, so add in buffer to data domain
  xScale.domain([d3.min(data, xValue)-1, d3.max(data, xValue)+1]);
  yScale.domain([d3.min(data, yValue)-1, d3.max(data, yValue)+1]);

  // x-axis
  svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
    .append("text")
      .attr("class", "label")
      .attr("x", width)
      .attr("y", -6)
      .style("text-anchor", "end")
      .text("TSNE Axis 0");

  // y-axis
  svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("class", "label")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("TSNE Axis 1");

    //title
  svg.append("text")
        .attr("x", (width / 2))             
        .attr("y", 0 - (margin.top / 2))
        .attr("text-anchor", "middle")  
        .style("font-size", "16px") 
        .style("text-decoration", "underline")  
        .text("Dimensionality Reduction of nickname={},k_type={},splitid={},quantile={},perfect={},even={}");

  // draw dots
  svg.selectAll(".dot")
      .data(data)
    .enter().append("circle")
      .attr("class", "dot")
      .attr("r", 3.5)
      .attr("cx", xMap)
      .attr("cy", yMap)
      .style("opacity", function(d) {{return d.correctness}})
      .style("fill", function(d) {{ return color(cValue(d));}}) 
      .on("click", function(d){{
         window.location = d["url"];
        }})
      .on("mouseover", function(d) {{
          tooltip.transition()
               .duration(200)
               .style("opacity", .9);
          tooltip.html(d.label + "," + d.cat_purity)
               .style("left", (d3.event.pageX + 5) + "px")
               .style("top", (d3.event.pageY - 28) + "px");
      }})
      .on("mouseout", function(d) {{
          tooltip.transition()
               .duration(500)
               .style("opacity", 0);
      }});

  // draw legend
  var legend = svg.selectAll(".legend")
      .data(color.domain())
    .enter().append("g")
      .attr("class", "legend")
      .attr("transform", function(d, i) {{ return "translate(0," + i * 20 + ")"; }});

  // draw legend colored rectangles
  legend.append("rect")
      .attr("x", width - 18)
      .attr("width", 18)
      .attr("height", 18)
      .style("fill", color);

  // draw legend text
  legend.append("text")
      .attr("x", width - 24)
      .attr("y", 9)
      .attr("dy", ".35em")
      .style("text-anchor", "end")
      .text(function(d) {{ return d;}})
}});

</script>
</body>
</html>
'''.format(fname,nickname,k_type,splitid,quantile,perfect,even)

def arch_script(nickname,splitid,quantile,perfect,even,fname):
    return '''
<!DOCTYPE html>
<html>
<meta charset="utf-8">

<style>
body {{
  font: 11px sans-serif;
}}

.axis path,
.axis line {{
  fill: none;
  stroke: #000;
  shape-rendering: crispEdges;
}}

.dot {{
  stroke: #000;
}}

.tooltip {{
  position: absolute;
  width: 200px;
  height: 28px;
  pointer-events: none;
}}
</style>
<body>
<script src="http://d3js.org/d3.v3.min.js"></script>

<script>
var margin = {{top: 20, right: 20, bottom: 30, left: 40}},
    width = 960 - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

/* 
 * value accessor - returns the value to encode for a given data object.
 * scale - maps value to a visual display encoding, such as a pixel position.
 * map function - maps from data value to display value
 * axis - sets up axis
 */ 

// setup x 
var xValue = function(d) {{ return d.tsnex;}}, // data -> value
    xScale = d3.scale.linear().range([0, width]), // value -> display
    xMap = function(d) {{ return xScale(xValue(d));}}, // data -> display
    xAxis = d3.svg.axis().scale(xScale).orient("bottom");

// setup y
var yValue = function(d) {{ return d.tsney;}}, // data -> value
    yScale = d3.scale.linear().range([height, 0]), // value -> display
    yMap = function(d) {{ return yScale(yValue(d));}}, // data -> display
    yAxis = d3.svg.axis().scale(yScale).orient("left");


// setup fill color
var cValue = function(d) {{ return d.cluster;}},
    color = d3.scale.category10();

// add the graph canvas to the body of the webpage
var svg = d3.select("body").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// add the tooltip area to the webpage
var tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

// load data
//
d3.csv("{}", function(error, data) {{

  // change string (from CSV) into number format
  data.forEach(function(d) {{
    d.tsnex = +d.tsnex;
    d.tsney = +d.tsney;
    d.cluster = +d.cluster;
    d.clust_purity = +d.clust_purity;
    d.cat_purity = +d.cat_purity;
    d.correctness = +d.correctness;
    d.label = d.label;
    d.url = d.url;
    //console.log(d.url);
  }});

  // don't want dots overlapping axis, so add in buffer to data domain
  xScale.domain([d3.min(data, xValue)-1, d3.max(data, xValue)+1]);
  yScale.domain([d3.min(data, yValue)-1, d3.max(data, yValue)+1]);

  // x-axis
  svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
    .append("text")
      .attr("class", "label")
      .attr("x", width)
      .attr("y", -6)
      .style("text-anchor", "end")
      .text("TSNE Axis 0");

  // y-axis
  svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("class", "label")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("TSNE Axis 1");

// title
  svg.append("text")
        .attr("x", (width / 2))             
        .attr("y", 0 - (margin.top / 2))
        .attr("text-anchor", "middle")  
        .style("font-size", "16px") 
        .style("text-decoration", "underline")  
        .text("Dimensionality Reduction of nickname={},splitid={},quantile={},perfect={},even={}");

  // draw dots
  svg.selectAll(".dot")
      .data(data)
    .enter().append("circle")
      .attr("class", "dot")
      .attr("r", 3.5)
      .attr("cx", xMap)
      .attr("cy", yMap)
      .style("opacity", function(d) {{return d.correctness}})
      .style("fill", function(d) {{ return color(cValue(d));}}) 
      .on("click", function(d){{
         window.location = d["url"];
        }})
      .on("mouseover", function(d) {{
          tooltip.transition()
               .duration(200)
               .style("opacity", .9);
          tooltip.html(d.label + "," + d.cat_purity)
               .style("left", (d3.event.pageX + 5) + "px")
               .style("top", (d3.event.pageY - 28) + "px");
      }})
      .on("mouseout", function(d) {{
          tooltip.transition()
               .duration(500)
               .style("opacity", 0);
      }});

  // draw legend
  var legend = svg.selectAll(".legend")
      .data(color.domain())
    .enter().append("g")
      .attr("class", "legend")
      .attr("transform", function(d, i) {{ return "translate(0," + i * 20 + ")"; }});

  // draw legend colored rectangles
  legend.append("rect")
      .attr("x", width - 18)
      .attr("width", 18)
      .attr("height", 18)
      .style("fill", color);

  // draw legend text
  legend.append("text")
      .attr("x", width - 24)
      .attr("y", 9)
      .attr("dy", ".35em")
      .style("text-anchor", "end")
      .text(function(d) {{ return d;}})
}});

</script>
</body>
</html>
'''.format(fname,nickname,splitid,quantile,perfect,even)


def mkpage(csvname,htmlname,variety,nickname,splitid,quantile,perfect,even,k_type=None):
    assert(variety in {'greedy','arch'}) 
    with open(htmlname,'w') as f:
        if variety == 'arch':
            f.write(arch_script(nickname,splitid,quantile,perfect,even,csvname))
        else:
            f.write(greedy_script(nickname,k_type,splitid,quantile,perfect,even,csvname))
    print("Wrote {}".format(htmlname))
    return

def doall( ):
    # I should get nice titles going.
    fnames = os.listdir('data/embed')
    for fname in fnames:
        rt,ext = os.path.splitext(fname)
        out = rt+'.html'
        lines = open(fname,'r').readlines()
        if len(lines) in [1,2]:
            subprocess.call(["rm",out]) 
            continue
        with open(out,'w') as f:
            if 'arch' in fname: 
                f.write(arch_script(fname)) 
            else:
                f.write(greedy_script(fname)) 

def webfmt(greedy:bool,nickname,splitid,num_clusters,num_samples,perfect,even,trial,k_type=None,root='/home/aseewald/public_html/',quantiles=[1.0]):
    for quantile in quantiles:
        if greedy:
            assert(k_type is not None)
            fmt = "greedy_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(nickname,k_type,splitid,quantile,num_clusters,num_samples,perfect,even,trial)
        else:
            assert(k_type is None)
            fmt = "arch_{}_{}_{}_{}_{}_{}_{}_{}".format(nickname,splitid,quantile,num_clusters,num_samples,perfect,even,trial)
        csvname = "data/embed/"+fmt + '.csv'
        htmlname = root + fmt + '.html'
        return {quantile : (csvname,htmlname) for quantile in quantiles}

