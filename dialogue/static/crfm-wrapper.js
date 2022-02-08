
const ChatBox = {
  data() {
    return {
      session_uuid: null,
      user_uuid: null,
      payload: null,
      utterances: [
      ],
      newUtterance: '',
    }
  },
  methods: {
    pushUtterance: function(speaker, text) {
      this.utterances.push({speaker: speaker, text: text});
      this.$nextTick(function(){
          realign_transcript()
      });
    },
    submit: function(){
      document.newUtteranceForm.newUtterance.focus();
      var that = this;
      if(!this.newUtterance=='') {
          this.pushUtterance('user', this.newUtterance);
      }
      axios.post('/conversation', {
        payload: this.payload,
        session_uuid: this.session_uuid,
        user_uuid: this.user_uuid,
        user_utterance: this.newUtterance
      })
          .then(function(response) {
            var data = response.data;
            that.pushUtterance('bot', data.bot_utterance);
            that.payload = data.payload;
            that.session_uuid= data.session_uuid;
            that.user_uuid = data.user_uuid;
            console.log(response);
          })
          .catch(function(error){
            console.log(error);
          });
      this.newUtterance= '';
    }

  }
}
function realign_transcript(){
    var element = document.getElementsByClassName("transcript");
    element[0].scrollTop = element[0].scrollHeight;
}
window.addEventListener('resize', realign_transcript);
const app = Vue.createApp(ChatBox);
const vm = app.mount("#app");
