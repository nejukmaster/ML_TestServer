const request = require('request');
const fs = require('fs');

var req = request.post("http://....:0000", function (err, resp, body) {
    if (err)
      console.log(err);
  });
var form = req.form();
form.append('file', fs.createReadStream("./img.png"));

console.log(form);