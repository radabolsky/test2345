//const inputElement = document.getElementById("input");
//inputElement.addEventListener("change", handleFiles, false);
//
//
//function handleFiles() {
//  const fileList = this.files; /* now you can work with the file list */
//}


//const selectedFile = document.getElementById('input').files[0];

function send_image_data(id_) {
    let formData = new FormData();
    let picture = document.getElementById('picture').files[0];

    color = $("input[name=scanner_type]:checked", "#color_radio").val();
    formData.append("color", color)

    size = $("#pictute_size").val();
    formData.append("size", size)

    formData.append("picture", picture);

    $.ajax({
        url: `/api/v1/color_definition/redirect`,
        cache: false,
        data: formData,
        contentType: false,
        processData: false,
        method: 'POST',
        success: function(response){
            console.log(response)
            $(`#${id_}`).show();
            $(`#${id_}`).append(`<a class="test form-control-label" href=${response.redirect} role="button">Go to results</a>`)
            }
        }
      );
}
