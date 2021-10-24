var rec

$(document).ready(function () {
    var timeInterval = NaN
    var count = 3

    a = "{\"models\":{\"acted_cnn_lstm\":{\"Anger\":0.1365242600440979,\"Disgust\":0.3891822099685669,\"Fear\":0.12726570665836334,\"Happy\":0.5330187678337097,\"Neutral\":0.11657091230154037,\"Sad\":0.1352783441543579},\"acted_cnn_lstm_attention\":{\"Anger\":0.059719838201999664,\"Disgust\":0.3120494782924652,\"Fear\":0.19841468334197998,\"Happy\":0.22142034769058228,\"Neutral\":0.2976442575454712,\"Sad\":0.3227205276489258},\"acted_cnn_lstm_attention_multitask\":{\"Anger\":0.14646202325820923,\"Disgust\":0.16858071088790894,\"Fear\":0.18085850775241852,\"Happy\":0.16637471318244934,\"Neutral\":0.2895333170890808,\"Sad\":0.3357485830783844,\"Male\":0.23208677768707275,\"Female\":0.7679132223129272,\"Caucasian\":0.9004549011588097,\"Non-Caucasian\":0.09954509884119034},\"observed_cnn_lstm\":{\"Anger\":0.20152424275875092,\"Disgust\":0.11315349489450455,\"Fear\":0.1294436752796173,\"Happy\":0.1721806824207306,\"Neutral\":0.6916586756706238,\"Sad\":0.12743154168128967},\"observed_cnn_lstm_attention\":{\"Anger\":0.0805794820189476,\"Disgust\":0.07485463470220566,\"Fear\":0.10949211567640305,\"Happy\":0.06691548973321915,\"Neutral\":0.9158254861831665,\"Sad\":0.09995155781507492},\"observed_cnn_lstm_attention_multitask\":{\"Anger\":0.06858907639980316,\"Disgust\":0.07232820987701416,\"Fear\":0.16383574903011322,\"Happy\":0.16118772327899933,\"Neutral\":0.8073530197143555,\"Sad\":0.06730818003416061,\"Male\":0.8586928695440292,\"Female\":0.14130713045597076,\"Caucasian\":0.926032654941082,\"Non-Caucasian\":0.073967345058918}}}";
    results = JSON.parse(a)
    models = results['models']
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
        $('p[value="time"]').text(count + " s")
        count--;
        timeInterval = setInterval(function () {
            $('p[value="time"]').text(count + " s")
            count--;
            if (count < 0) {
                // Stop
                clearInterval(timeInterval)
                count = 3
                isRunning = false
                $('p[value="time"]').html("Ready! &#128540;")
                rec.stop()
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
    $.post({
        url: "/ad",
        data: formData,
        processData: false,
        contentType: false,
        success: function (e) {
            isRunning = false
            results = JSON.parse(e)
            models = results['models']
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
        },
        error: function (e) {
            isRunning = false
            showToast("Failed")
        }
    });
}

function onload() {
    if (navigator.mediaDevices === undefined) {
        showToast("Your browser is not supported")
        return
    }
    var audio_seq = []
    navigator.mediaDevices.getUserMedia({ video: false, audio: true }).then(stream => {
        window.localStream = stream;
        rec = new MediaRecorder(stream, { mineType: "audio/wav" });
        console.error(rec)
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
            var blob = new Blob(audio_seq, { 'type': 'audio/ogg; codecs=opus' });
            source.src = window.URL.createObjectURL(blob);

            audio.load();
            // postdata(e.data)
        }
    }).catch(err => {
        console.log("u got an error:" + err)
    });
}

function getHighestScore(scores) {
    score = 0
    key = ''
    jQuery.each(scores, function (i, val) {
        if (i == 'Caucasian' || i == 'Non-Caucasian' || i == "Female" || i == "Male") {
            return
        }
        if (score < val) {
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