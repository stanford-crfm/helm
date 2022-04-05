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
          text: "Please rate how consistent your partner sounds",
          type: "likert",
          options: ["1 - Always inconsistent", "2 - Often inconsistent", "3 - Inconsistent half of the time", "4 - Occasionally inconsistent", "5 - Always consistent"]
        },
        {
          tag: "feedback",
          text: "Is there anything else you would like to say about the conversation?",
          type: "freeForm",
        },
        /*{
          tag: "humanness",
          text: "Please rate how human your partner sounds (1: Very inhuman, 5: Very human)",
          type: "likert",
        },
        {
          tag: "quality",
          text: "Please rate the overall quality of the conversation (1: Very inhuman, 5: Very human)",
          type: "likert",
        }*/
      ];
    for (let question of questions) {
      if (question.type === "likert") {
        question.rating = "";
      } else if (question.type === "turnSelection") {
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
      prompt: "",
      optedout: false,
      //utterances: [],
      utterances: [
        /*{speaker: "user", text: "hi"},
        {speaker: "bot", text: "hi"},
        {speaker: "user", text: "what are you up to?"},
        {speaker: "bot", text: "nothing, just watching tv"},*/

      ],
      newUtterance: "",
      //isConversationOver: true,
      isConversationOver: false,
      currentQuestionIdx: -1,
      error: false,
      success: false,
    }
  },
  /*watch: {
    "currentQuestion.notaUtterance"(newValue){
      if (newValue){
        currentQuestion.selectedUtterances = [];
      }
    }
  },*/
  computed: {
    isInterviewOver: function () {
      return this.currentQuestionIdx === this.questions.length - 1;
    },
    currentQuestion: function () {
      if (this.currentQuestionIdx < 0) {
        return {
          tag: "Dummy",
          text: "Dummy",
          type: "Dummy",
          selectedUtterances: [],
          notaUtterance: false,
          rating: ""
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
      if (!(this.newUtterance === "")) {
        this.pushUtterance("user", this.newUtterance);
      }
      axios.post("/api/dialogue/conversation", {
        payload: this.payload,
        session_uuid: this.session_uuid,
        user_uuid: this.user_uuid,
        user_utterance: this.newUtterance,
        optedout: this.optedout,
      })
        .then(function (response) {
          var data = response.data;
          that.pushUtterance("bot", data.bot_utterance);
          that.payload = data.payload;
          that.session_uuid = data.session_uuid;
          that.user_uuid = data.user_uuid;
          console.log(response);
        })
        .catch(function (error) {
          console.log(error);
        });
      this.newUtterance = "";
    },
    toggleSelectedUtterance: function (idx) {
      if (this.isConversationOver && this.currentQuestionIdx > -1 && this.currentQuestion.type === "turnSelection") {
        //if(!"selectedUtterances" in this.currentQuestion) {this.currentQuestion.selectedUtterances=[]}
        this.currentQuestion.selectedUtterances = _.xor(this.currentQuestion.selectedUtterances, [idx])
        if (this.currentQuestion.selectedUtterances.length > 0) {
          this.currentQuestion.notaUtterance = false;
        }
      }
    },
    toggleNotaUtterance: function (event) {
      this.currentQuestion.notaUtterance = !this.currentQuestion.notaUtterance;
      console.log(this.currentQuestion.notaUtterance);
      this.currentQuestion.selectedUtterances = [];
    },
    endConversation: function () {
      this.isConversationOver = true;
      this.currentQuestionIdx = 0;
    },
    submitAnswers: function () {
      var that = this;
      if (this.validate() || this.optedout) {
        axios.post("/api/dialogue/interview", {
          payload: this.payload,
          session_uuid: this.session_uuid,
          user_uuid: this.user_uuid,
          user_utterance: this.newUtterance,
          utterances: this.utterances,
          questions: this.questions,
          optedout: this.optedout,
        })
          .then(function (response) {
            that.success = true;
            that.error = false;
            console.log("success");
          })
          .catch(function (error) {
            console.log(error);
            that.error = error;
            that.success = false;
          });
      }
      //TODO: AJAX request to server
    },
    nextQuestion: function () {
      if (this.validate()) {
        this.error = false;
        this.currentQuestionIdx += 1;
      }
    },
    prevQuestion: function () {
      this.currentQuestionIdx -= 1;
    },
    validate: function () {
      if (this.currentQuestion.type === "likert") {
        if (!(this.currentQuestion.options.includes(this.currentQuestion.rating))) {
          this.error = "Pick one out of the given options";
          return false;
        }
        return true
      }
      else if (this.currentQuestion.type === "turnSelection") {
        if (this.currentQuestion.selectedUtterances.length === 0 && !this.currentQuestion.notaUtterance) {
          this.error = "Either select one or more utterances or check 'None of the utterances'";
          return false;
        }
        return true;
      }
      else if (this.currentQuestion.type === "freeForm") {
        return true;
      }
      else {
        console.log("Question type not defined")
      }
    },
    optout: function () {
      this.optedout = true;
      if (!this.isConversationOver) { this.endConversation(); }
      this.submitAnswers();
    }
  }
}
function realign_transcript() {
  var element = document.getElementsByClassName("transcript");
  element[0].scrollTop = element[0].scrollHeight;
}
window.addEventListener("resize", realign_transcript);
const app = Vue.createApp(ChatBox);
const vm = app.mount("#app");
axios.post("/api/dialogue/start", {
})
  .then(function (response) {
    vm.prompt = response.data.prompt;
    console.log(vm.prompt);
  })
  .catch(function (error) {
    console.log(error);
  });
$(window).on('load', function () {
  $('#consentModal').modal('show');
});