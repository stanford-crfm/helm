

//v-for="(utterance, index) in utterances
const ChatBox = {
  data() {
      var questions = 
        [
          {
            tag: "consistency-utterances",
            text: "Select the utterances where your partner sounds inconsistent",
            type: "turnSelection",
          },
          {
            tag: "consistency-rating",
            text: "Please rate how consistent your partner sounds \n\n (1: Very consistent, 5: Very inconsistent)",
            type: "likert",
          },
          {
            tag: "humanness",
            text: "Please rate how human your partner sounds (1: Very inhuman, 5: Very human)",
            type: "likert",
          },
          {
            tag: "quality",
            text: "Please rate the overall quality of the conversation (1: Very inhuman, 5: Very human)",
            type: "likert",
          }
        ];
      for (let question of questions){
        if(question.type == 'likert'){
          question.rating = '';
        } else if (question.type=='turnSelection'){
          question.selectedUtterances = [];
          question.notaUtterance = false;
        }
      }
      console.log(questions);
    return {
      session_uuid: null,
      user_uuid: null,
      payload: null,
      questions: questions,
      //utterances: [],
      utterances: [
        {speaker: "user", text: "hi"},
        {speaker: "bot", text: "hi"},
        {speaker: "user", text: "what are you up to?"},
        {speaker: "bot", text: "nothing, just watching tv"},

      ],
      newUtterance: '',
      //isConversationOver: true,
      isConversationOver: false,
      currentQuestionIdx: -1,
    }
  },
  computed:{
    isInterviewOver: function(){
      return this.currentQuestionIdx == this.questions.length-1;
    },
    currentQuestion: function(){
      if (this.currentQuestionIdx<0){
        return {
          tag: "Dummy",
          text: "Dummy",
          type: "Dummy",
          selectedUtterances: [],
          notaUtterance: false,
          rating: ''
        }
      }
      return this.questions[this.currentQuestionIdx]
    }
  },
  methods: {
    pushUtterance: function (speaker, text) {
      this.utterances.push({ speaker: speaker, text: text });
      this.$nextTick(function () {
        realign_transcript()
      });
    },
    submit: function () {
      document.newUtteranceForm.newUtterance.focus();
      var that = this;
      if (!this.newUtterance == '') {
        this.pushUtterance('user', this.newUtterance);
      }
      axios.post('/conversation', {
        payload: this.payload,
        session_uuid: this.session_uuid,
        user_uuid: this.user_uuid,
        user_utterance: this.newUtterance
      })
        .then(function (response) {
          var data = response.data;
          that.pushUtterance('bot', data.bot_utterance);
          that.payload = data.payload;
          that.session_uuid = data.session_uuid;
          that.user_uuid = data.user_uuid;
          console.log(response);
        })
        .catch(function (error) {
          console.log(error);
        });
      this.newUtterance = '';
    },
    toggleSelectedUtterance: function(idx){
      if(this.isConversationOver && this.currentQuestionIdx>-1 && this.currentQuestion.type=='turnSelection'){
        //if(!'selectedUtterances' in this.currentQuestion) {this.currentQuestion.selectedUtterances=[]}
        this.currentQuestion.selectedUtterances = _.xor(this.currentQuestion.selectedUtterances, [idx])
      }
    },
    endConversation: function(){
      this.isConversationOver = true;
      this.currentQuestionIdx = 0;
    },
    submitAnswers: function(){
      //TODO: AJAX request to server
    },
    nextQuestion: function(){
      this.currentQuestionIdx+=1
    },
    prevQuestion: function(){
      this.currentQuestionIdx-=1
    },
    saveCurrentAnswer: function(){
      this.answers[this.currentQuestionIdx]
    }

  }
}
function realign_transcript() {
  var element = document.getElementsByClassName("transcript");
  element[0].scrollTop = element[0].scrollHeight;
}
window.addEventListener('resize', realign_transcript);
const app = Vue.createApp(ChatBox);
const vm = app.mount("#app");
