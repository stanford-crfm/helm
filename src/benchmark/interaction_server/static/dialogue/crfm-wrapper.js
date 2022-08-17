//v-for="(utterance, index) in utterances
const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);
const ChatBox = {
	data() {
		var questions =
			[	
				{
					tag: "experience",
					text: "How would you describe your experience chatting with this bot?",
					type: "freeForm",
				},
				{
					tag: "experience-positive",
					text: "What did you like about the experience?",
					type: "freeForm",
				},
				{
					tag: "experience-negative",
					text: "What did you not like about the experience?",
					type: "freeForm",
				},
				{
					tag: "additional-questions",
					text: "What questions would you have liked to be asked?",
					type: "freeForm",
				}
				/*
				{
					tag: "workerID",
					text: "Please enter your Mechanical Turk WorkerID for payment processing",
					type: "freeForm",
				},
				{
					tag: "interestingness-utterances",
					text: "Mark the responses that were particularly interesting or boring",
					type: "turn-ternary",
					options: {positive: "interesting" , neutral: "normal", negative: "boring"}
				},
				{
					tag: "preference-utterances",
					text: "Which responses made you NOT want to talk with the chatbot again?",
					type: "turn-binary",
				},
				{
					tag: "humanness-utterances",
					text: "Which responses did NOT sound human?",
					type: "turn-binary",
				}, 
				{
					tag: "sensibility-utterances",
					text: "Mark responses where the chatbot did NOT make sense",
					type: "turn-binary",
				},
				{
					tag: "specificity-utterances",
					text: "Mark the responses that were NOT specific to what you had said, i.e. responses that could have been used in many different situations.  \n\n For example, if you say \"I love tennis\" then \"That’s nice\" would be a non-specific response, but \"Me too, I can’t get enough of Roger Federer!\" would be a specific response.",
					type: "turn-binary",
				},
				
				{
					tag: "positive-feedback-utterances",
					text: "If you gave the chatbot feedback (e.g. by making a correction or clarification), mark turns where it improved based on that feedback",
					type: "turn-binary",

				},
				{
					tag: "negative-feedback-utterances",
					text: "If you gave the chatbot feedback (e.g. by making a correction or clarification), mark turns where it did NOT improve based on that feedback",
					type: "turn-binary",
				},
				{
					tag: "making-up-utterances",
					text: "Mark responses where you felt that the chatbot was making things up",
					type: "turn-binary"
				},
				
				{
					tag: "consistency-utterances",
					text: "Which responses contradicted previous dialogue?",
					type: "turn-binary"
				},
				{
					tag: "tangents-utterances",
					text: "Mark responses where the chatbot went on a tangent",
					type: "turn-binary"
				},
				{
					tag: "contribution-utterances",
					text: "Mark the responses where the chatbot seemed particularly active or passive in the conversation",
					type: "turn-ternary",
					options: {positive: "active" , neutral: "normal", negative: "passive"}
				},
				{
					tag: "contribution-overall",
					text: "How much did the chatbot contribute to the conversation's direction?",
					type: "likert",
					options: ["1 - Not enough", "2 - The right amount", "3 - Too much"]
				}, 
				{
					tag: "quality-overall",
					text: "Would you want to talk to this chatbot again?",
					type: "likert",
					options: ["1 - No", "2 - Probably not", "3 - Maybe", "4 - Probably yes", "5 - Yes"]
				},
				{
					tag: "feedback",
					text: "Is there anything else you would like to say about the conversation?",
					type: "freeForm",
				}*/
			];
		for (let question of questions) {
			if (question.type === "likert") {
				question.rating = "";
			} else if (question.type === "turn-binary") {
				question.selectedUtterances = [];
				question.notaUtterance = false;
			} else if (question.type === "turn-ternary"){
				question.notaUtterance = false;
				question.turnAnnotations = [];
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
			utterances: [],
			newUtterance: "",
			isConversationOver: false,
			currentQuestionIdx: -1,
			error: false,
			success: false,
		}
	},
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
		},
		nWords: function() {
			var nWords = 0;
			for(utterance of this.utterances) {
				words = utterance["text"].split(" ");
				nWords = nWords + words.length;
			}
			return nWords;
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
				interaction_trace_id: urlParams.get("interaction_trace_id"), 
				run_name: urlParams.get("run_name"),
				user_id: urlParams.get("user_id"),
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
			if (this.isConversationOver && this.currentQuestionIdx > -1 && this.currentQuestion.type === "turn-binary") {
				//if(!"selectedUtterances" in this.currentQuestion) {this.currentQuestion.selectedUtterances=[]}
				this.currentQuestion.selectedUtterances = _.xor(this.currentQuestion.selectedUtterances, [idx])
				if (this.currentQuestion.selectedUtterances.length > 0) {
					this.currentQuestion.notaUtterance = false;
				}
			}
		},
		ternaryLabel: function (idx, label) {
			if (this.isConversationOver && this.currentQuestionIdx > -1 && this.currentQuestion.type === "turn-ternary") {
				this.currentQuestion.turnAnnotations[idx] = label; 
				if (!this.currentQuestion.turnAnnotations.every( function (ele) {return ele === "neutral"})) {
					this.currentQuestion.notaUtterance = false;
				}
			}
		},
		toggleNotaUtterance: function (event) {
			this.currentQuestion.notaUtterance = !this.currentQuestion.notaUtterance;
			console.log(this.currentQuestion.notaUtterance);
			if (this.currentQuestion.type === 'turn-binary'){
				this.currentQuestion.selectedUtterances = [];
			}
			else if (this.currentQuestion.type === 'turn-ternary'){
				var that = this;
				this.utterances.forEach(function (utt, idx){
					if (utt["speaker"] === "bot") {
						that.currentQuestion.turnAnnotations[idx] = "neutral"; 
					}
				});
			}
		},
		prepSurvey: function (response) {
			this.addDatasetQuestion(response);
			this.initSurveyResponses();
			this.isConversationOver = true;
			this.currentQuestionIdx = 0;
		}, 
		endConversation: function () {
			var that = this;
			axios.post("/api/dialogue/end", {
				interaction_trace_id: urlParams.get("interaction_trace_id"), 
				run_name: urlParams.get("run_name"),
				user_id: urlParams.get("user_id"),
				payload: this.payload,
				session_uuid: this.session_uuid,
				user_uuid: this.user_uuid,
			})
				.then(function (response) {
					that.prepSurvey(response)
					that.success = true;
					that.error = false;
					console.log("success");
				})
				.catch(function (error) {
					console.log(error);
					that.error = error;
					that.code = "dinosaur";
					that.success = false;
				});
		},
		initSurveyResponses: function(){
			for (let question of this.questions) {
				if (question.type === "likert") {
					question.rating = "";
				} else if (question.type === "turn-binary") {
					question.selectedUtterances = [];
					question.notaUtterance = false;
				} else if (question.type === "turn-ternary"){
					question.notaUtterance = false;
					var that = this;
					this.utterances.forEach(function (utt, idx){
						if (utt["speaker"] === "bot") {
							question.turnAnnotations[idx] = "neutral"; 
						}
					});
				}
			}
		},
		addDatasetQuestion: function(response){
			dataset_questions = {
				"benchmark.dialogue_scenarios.EmpatheticDialoguesScenario": 
				"Which responses did you feel an emotional connection to?",
				"benchmark.dialogue_scenarios.WizardOfWikipediaScenario": 
				"Which responses were informative?",
				"benchmark.dialogue_scenarios.CommonSenseScenario": 
				"Which responses made you feel the chatbot understood social contexts and situations?"
			};
			question = {
				tag: "dataset-specific",
				text: dataset_questions[response.data.scenario.class_name],
				type: "turn-binary",
			};
			this.questions.splice(-2, 0, question);
		},
		submitAnswers: function () {
			var that = this;
			if (this.validate() || this.optedout) {
				axios.post("/api/dialogue/interview", {
					interaction_trace_id: urlParams.get("interaction_trace_id"), 
					run_name: urlParams.get("run_name"),
					user_id: urlParams.get("user_id"),
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
						that.code = response.data.code;
						console.log("success");
					})
					.catch(function (error) {
						console.log(error);
						that.error = error;
						that.code = "dinosaur";
						that.success = false;
					});
			}
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
			else if (this.currentQuestion.type === "turn-binary") {
				if (this.currentQuestion.selectedUtterances.length === 0 && !this.currentQuestion.notaUtterance) {
					this.error = "Either select one or more utterances or check 'None of the utterances'";
					return false;
				}
				return true;
			}
			else if (this.currentQuestion.type === "turn-ternary"){
				if (this.currentQuestion.turnAnnotations.every(
					function (ele) {return ele === "neutral"}
				) && !this.currentQuestion.notaUtterance ) 
				{
					this.error = "Have you considered all turns and found that they were neither "+this.currentQuestion.options['positive']+" nor "+ this.currentQuestion.options['negative']+"? If so check 'All utterances are "+this.currentQuestion.options['neutral']; 
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
	interaction_trace_id: urlParams.get("interaction_trace_id"), 
	run_name: urlParams.get("run_name"),
	user_id: urlParams.get("user_id"),
})
	.then(function (response) {
		console.log(response.data);
		vm.prompt = response.data.display_prompt;
		for (let utt of response.data.utterances){
			vm.pushUtterance(utt.speaker, utt.text);
		}
		if(response.data.isConversationOver){
			vm.prepSurvey(response)
		}
		if(response.data.survey){
			console.log(response.data.survey);
			console.log(vm.questions);
			vm.questions=response.data.survey;
		}
		console.log(vm.prompt);
		console.log(vm.utterances);
	})
	.catch(function (error) {
		console.log(error);
	});
$(window).on('load', function () {
	$('#consentModal').modal('show');
});
