var rec
var isCommercialEnabled = true
var audio_seq = []
$(document).ready(function () {
    var timeInterval = NaN
    var count = 3
    onload()
    var isRunning = false
    $('a[value="trigger"]').on('click', function () {
        if (isRunning) {
            return
        }
        if (!rec) {
            showToast("Please check your microphone permission")
            return
        }
        isRunning = true
        audio_seq = []
        $('p[value="time"]').text(count + " s")
        count--;
        timeInterval = setInterval(function () {
            $('p[value="time"]').text(count + " s")
            count--;
            if (count < 0) {
                // Stop
                clearInterval(timeInterval)
                count = 3
                rec.stop()
                isRunning = false
                $('p[value="time"]').html("Processing...")
            }
        }, 1000)
        rec.start()
    });
});
var toastTimer
function showToast(text) {
    if (!toastTimer) {
        clearTimeout(toastTimer)
    }
    $('span[class=toast]').text(text)
    $('span[class=toast]').show()
    toastTimer = setTimeout(() => {
        $('span[class=toast]').hide()
    }, 2000);
}

function postdata(data) {
    var formData = new FormData();
    formData.append("audio", data, "audio.ogg")
    formData.append("isCommercialEnabled", '' + isCommercialEnabled)
    formData.append('br', rec.audioBitsPerSecond)
    $.post({
        url: "/ad",
        data: formData,
        processData: false,
        contentType: false,
        success: function (e) {
            console.log(e)
            isRunning = false
            results = JSON.parse(e)
            models = results['models']
            commercial_models = results['commercials']
            jQuery.each(models, function (i, val) {
                list = getHighestScore(val)
                var final = getEmotionByKey(list[0]) + ': ' + list[1]
                var ca = isCaucasian(val)
                if (ca != null) {
                    final += ', ' + getEmotionByKey(ca)
                }
                var gender = getGender(val)
                if (gender != null) {

                    final += ', ' + getEmotionByKey(gender)
                }
                $('p[value="' + i + '"]').html(final)
            });
            $('p[value="time"]').html("Ready! &#128540;")
            if (commercial_models != null) {
                updateVokaturi(getHighestScore(commercial_models["vokaturi"]))
                updateEmpath(getHighestScore(commercial_models["empath"]))
                updateDeepAffects(commercial_models["deepaffect"])
            }
        },
        error: function (e) {
            isRunning = false
            $('p[value="time"]').html("Ready! &#128540;")
            showToast("Failed")
        }
    });
}

function updateVokaturi(scores) {
    key = scores[0]
    score = ': ' + scores[1]

    var txt = ''
    switch (key) {
        case "Angry": txt = "&#128545; Angry"; break;
        case "Disgust": txt = "&#128530; Disgust"; break;
        case "Fear": txt = "&#128561; Fear"; break;
        case "Happy": txt = "&#128541; Happy"; break;
        case "Neutral": txt = "&#128528; Neutral"; break;
        default: txt = 'I don\'t know'; score = '';
    }

    $('p[value="vokaturi"]').html(txt + score)

}
function updateEmpath(scores) {
    key = scores[0]
    score = ': ' + scores[1]
    var txt = ''
    switch (key) {
        case "calm": txt = "&#128528; Calm"; break;
        case "anger": txt = "&#128545; Calm"; break;
        case "joy": txt = "&#128521; Joy"; break;
        case "sorrow": txt = "&#128534; Sorrow"; break;
        case "energy": txt = "&#128513; Energy"; break;
        default: txt = 'I don\'t know'; score = '';
    }
    $('p[value="empath"]').html(txt + score)

}
function updateDeepAffects(scores) {
    key = ''
    if (scores != null && scores.length > 0) {
        txt = scores[0]
    }

    switch (key) {
        case "anger": txt = "&#128545; Anger"; break;
        case "neutral": txt = "&#128528; Neutral"; break;
        case "excited": txt = "&#128525; Excited"; break;
        case "frustration": txt = "&#128534; Frustration"; break;
        case "happy": txt = "&#128541; Happy"; break;
        case "sad": txt = "&#128557; Sad"; break;
        default: txt = 'I don\'t know';
    }

    $('p[value="deepaffects"]').html(txt)
}

function onload() {
    if (navigator.mediaDevices === undefined) {
        showToast("Your browser is not supported")
        return
    }
    navigator.mediaDevices.getUserMedia({ video: false, audio: true }).then(stream => {

        window.localStream = stream;
        rec = new MediaRecorder(stream, { mineType: "audio/wav" });

        console.log(rec.audioBitsPerSecond)
        rec.ondataavailable = e => {
            if (e.data === undefined) {
                return
            }
            audio_seq.push(e.data)
        };
        rec.onstop = function (e) {
            var audio = document.getElementById('audio');
            audio.controls = true;
            var source = document.getElementById('audioSource');
            source.src = null
            var blob = new Blob(audio_seq, { 'type': 'audio/ogg; codecs=opus' });
            source.src = window.URL.createObjectURL(blob);

            audio.load();
            postdata(blob)
        }
    }).catch(err => {
        console.log("u got an error:" + err)
    });
    $('input[name=ck_commerical]').prop('checked', true)
    $('input[name=ck_commerical]').change(function (e) {
        if (this.checked) {
            showToast("Commericial Models are enabled")
            isCommercialEnabled = true
            updateCommercialResults('empath', 'Enabled', false)
            updateCommercialResults('vokaturi', 'Enabled', false)
            updateCommercialResults('deepaffects', 'Enabled', false)
        } else {
            isCommercialEnabled = false
            showToast("Commericial Models are disabled")
            updateCommercialResults('empath', 'Disabled', true)
            updateCommercialResults('vokaturi', 'Disabled', true)
            updateCommercialResults('deepaffects', 'Disabled', true)
        }
    })
}

function updateCommercialResults(name, scores, state) {
    $('p[value=' + name + ']').html(scores)
    $('p[value=' + name + ']').prop('disable', state)
    if (state) {
        $('p[value=' + name + ']').css('color', '#5072A7 ')
    } else {

        $('p[value=' + name + ']').css('color', 'cornsilk')
    }
}

function getHighestScore(scores) {
    score = 0
    key = ''
    jQuery.each(scores, function (i, val) {
        if (i == 'Caucasian'
            || i == 'Non-Caucasian'
            || i == "Female"
            || i == "Male"
            || i == 'error') {
            return
        }
        if (score <= val) {
            score = val
            key = i
        }
    });

    return [key, score.toFixed(2)]
}

function isCaucasian(scores) {
    if (scores["Caucasian"] === undefined) {
        return null
    }
    return scores["Caucasian"] > scores["Non-Caucasian"] ? "Caucasian" : "Non-Caucasian"
}

function getGender(scores) {
    if (scores["Female"] === undefined) {
        return null
    }
    return scores["Female"] > scores["Male"] ? "Female" : "Male"
}

function getEmotionByKey(key) {
    switch (key) {
        case "Anger": return "&#128545; Anger"
        case "Disgust": return "&#128530; Disgust"
        case "Fear": return "&#128561; Fear"
        case "Happy": return "&#128541; Happy"
        case "Neutral": return "&#128528; Neutral"
        case "Sad": return "&#128557; Sad"
        case "Male": return "&#129333; Male"
        case "Female": return "&#129501; Female"
        case "Caucasian": return "&#129340; Caucasian"
        case "Non-Caucasian": return "&#128694; Non-Caucasian"
        default: return 'I don\'t know'
    }
}