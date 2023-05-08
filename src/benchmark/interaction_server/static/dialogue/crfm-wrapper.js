//v-for="(utterance, index) in utterances
const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);
const ChatBox = {
	data() {
		var questions = [	
			{
									tag: "workerID",
									text: "Please enter your Mechanical Turk WorkerID for payment processing",
									type: "freeForm",
							},
			{
				tag: "initiative-1",
				type: "combo",
				text: "",
				likert: {   
							text: "How much do you agree with the following statement: \n\n 'The chatbot did enough to keep the conversation going'",
							options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
						},
				turn: true,
				why: {
							options: ["the bot was responsive to me", "the bot shared information", "the bot asked the right questions",
										"the bot was unresponsive to me", "the bot gave close-ended answers", "the bot did <b> not </b> ask enough questions", 
										 "the bot asked the wrong questions"]
						}
			},			
			{
				tag: "initiative-2",
				type: "combo",
				text: "",
				likert: {	
					text: "How much do agree with the following statement: 'I had enough opportunities to drive the conversation/choose topics'",
					options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
				},
				turn: true,
				why: {
					options: ["the bot asked enough questions about topics I chose", "the bot let me choose topics",
						 "the bot did not ask enough questions about topics I chose", "the bot chose too many topics",
							 "the bot let me drive the conversation enough", "the bot did not let me drive the conversation enough"]
				}
			},
			{
				tag: "interested",
				type: "combo",
				text: "",
				likert: {	
					text: "Did the chatbot seem interested in you?",
					options: ["1 - Not enough", "2 - The right amount", "3 - Too much"]
				},
				turn: true,
				why: {
					options: ["the bot asked questions about me or my interests", "I felt like the bot was listening to me", "the bot followed through on the topics I raised",
					"the bot did not ask enough questions about me or my interests", "I did not feel like the bot was listening to me", "the bot gave generic responses to what I said"]
				}
			},
			{
				tag: "sharing",
				type: "combo",

				text: "",
				likert: {	
					text: "Did the chatbot share about itself?",
					options: ["1 - Not enough", "2 - The right amount", "3 - Too much"]
				},
				turn: true,
				why: {
					options: ["the bot shared about itself", "the bot was genuine", "the bot did not share about itself", "the bot was inappropriate"]
				}
			},
			{
				tag: "fun",
				type: "combo",
				text: "",
				likert: {
					text: "How much do you agree with the following statement: 'Talking to the chatbot was fun.'",
					options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"],
				},
				turn: true,
				why: {
					options: ["the bot was fun", "the bot had a good sense of humor", "I was interested in what the bot was saying", "the bot was boring/bland", "the bot was not engaging", "I got tired of talking to the bot"]
				}
			},
			{
				tag: "emotional-connection",
				type: "combo",
				text: "",
				likert: {
					text: "How much do you agree with the following statement: “I felt an emotional connection to the chatbot.”  For example, the chatbot understood how you were feeling, or made you feel less lonely.",
					options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
				},
				turn: true,
				why: {
					options: ["I felt like I was talking to a friend", "the bot seemed human", "the bot made me feel less lonely",
													"the bot understood my feelings", "the bot did not like me", "the bot made me feel uncomfortable", "I did not like the bot"]
				}
			},
			{
				tag: "communication-skills",
				type: "combo",
				text: "",
				likert: {
					text: "How much do you agree with the following statement: “The chatbot displayed good communication skills” For example, the bot was coherent, did not repeat itself, understood what I was saying, and was logically correct.",
					options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
				},
				turn: true,
				why: {
					options: ["I understood everything the bot was talking about", "what the bot said was always consistent with the scenario", "I could not understand the bot's language", "the bot repeated itself", "the bot said things that were inconsistent with the scenario", "the bot contradicted itself", "I understood what the bot was saying, but it was absurd/ridiculous", "there were times when the bot did not make sense"]
				}
			},
			{
				tag: "social-skills",
				type: "combo",
				text: "",
				likert: {
					text: "How much do you agree with the following statement: “The chatbot responded in a way that was socially appropriate” For example, the bot was polite or the bot understood social situations.",
					options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
				},
				turn: true,
				why: {
					options: ["the bot was polite", "the bot understood social situations", "the bot was rude", "the bot was inappropriate", "the bot was argumentative", "the bot was uncooperative", "the bot was too agreeable"]
				}
			},
			{
									tag: "feedback-bot",
									text: "Is there anything else you would like to say about the bot?",
									type: "freeForm",
							},
			{
									tag: "feedback-survey",
									text: "Is there anything else you would like to say about the survey?",
									type: "freeForm",
							}
			
		];
		/*var questions =
			{"crfm-all": [	
				{
                                        tag: "workerID",
                                        text: "Please enter your Mechanical Turk WorkerID for payment processing",
                                        type: "freeForm",
                                },
				{
					tag: "initiative-1",
					type: "combo",
					text: "",
					likert: {   
								text: "How much do you agree with the following statement: \n\n 'The chatbot did enough to keep the conversation going'",
								options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
							},
					turn: true,
					why: {
								options: ["the bot was responsive to me", "the bot shared information", "the bot asked the right questions",
											"the bot was unresponsive to me", "the bot gave close-ended answers", "the bot did <b> not </b> ask enough questions", 
											 "the bot asked the wrong questions"]
							}
				},			
				{
					tag: "initiative-2",
					type: "combo",
					text: "",
					likert: {	
						text: "How much do agree with the following statement: 'I had enough opportunities to drive the conversation/choose topics'",
						options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
					},
					turn: true,
					why: {
						options: ["the bot asked enough questions about topics I chose", "the bot let me choose topics",
							 "the bot did not ask enough questions about topics I chose", "the bot chose too many topics",
						         "the bot let me drive the conversation enough", "the bot did not let me drive the conversation enough"]
					}
				},
				{
					tag: "interested",
					type: "combo",
					text: "",
					likert: {	
						text: "Did the chatbot seem interested in you?",
						options: ["1 - Not enough", "2 - The right amount", "3 - Too much"]
					},
					turn: true,
					why: {
						options: ["the bot asked questions about me or my interests", "I felt like the bot was listening to me", "the bot followed through on the topics I raised",
						"the bot did not ask enough questions about me or my interests", "I did not feel like the bot was listening to me", "the bot gave generic responses to what I said"]
					}
				},
				{
					tag: "sharing",
					type: "combo",

					text: "",
					likert: {	
						text: "Did the chatbot share about itself?",
						options: ["1 - Not enough", "2 - The right amount", "3 - Too much"]
					},
					turn: true,
					why: {
						options: ["the bot shared about itself", "the bot was genuine", "the bot did not share about itself", "the bot was inappropriate"]
					}
				},
				{
					tag: "fun",
					type: "combo",
					text: "",
					likert: {
						text: "How much do you agree with the following statement: 'Talking to the chatbot was fun.'",
						options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"],
					},
					turn: true,
					why: {
						options: ["the bot was fun", "the bot had a good sense of humor", "I was interested in what the bot was saying", "the bot was boring/bland", "the bot was not engaging", "I got tired of talking to the bot"]
					}
				},
				{
					tag: "emotional-connection",
					type: "combo",
					text: "",
					likert: {
						text: "How much do you agree with the following statement: “I felt an emotional connection to the chatbot.”  For example, the chatbot understood how you were feeling, or made you feel less lonely.",
						options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
					},
					turn: true,
					why: {
						options: ["I felt like I was talking to a friend", "the bot seemed human", "the bot made me feel less lonely",
                                                        "the bot understood my feelings", "the bot did not like me", "the bot made me feel uncomfortable", "I did not like the bot"]
					}
				},
				{
					tag: "communication-skills",
					type: "combo",
					text: "",
					likert: {
						text: "How much do you agree with the following statement: “The chatbot displayed good communication skills” For example, the bot was coherent, did not repeat itself, understood what I was saying, and was logically correct.",
						options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
					},
					turn: true,
					why: {
						options: ["I understood everything the bot was talking about", "what the bot said was always consistent with the scenario", "I could not understand the bot's language", "the bot repeated itself", "the bot said things that were inconsistent with the scenario", "the bot contradicted itself", "I understood what the bot was saying, but it was absurd/ridiculous", "there were times when the bot did not make sense"]
					}
				},
				{
					tag: "social-skills",
					type: "combo",
					text: "",
					likert: {
						text: "How much do you agree with the following statement: “The chatbot responded in a way that was socially appropriate” For example, the bot was polite or the bot understood social situations.",
						options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
					},
					turn: true,
					why: {
						options: ["the bot was polite", "the bot understood social situations", "the bot was rude", "the bot was inappropriate", "the bot was argumentative", "the bot was uncooperative", "the bot was too agreeable"]
					}
				},
				{
                                        tag: "feedback-bot",
                                        text: "Is there anything else you would like to say about the bot?",
                                        type: "freeForm",
                                },
				{
                                        tag: "feedback-survey",
                                        text: "Is there anything else you would like to say about the survey?",
                                        type: "freeForm",
                                }
				
			],
		"crfm-why":  [	
			{
									tag: "workerID",
									text: "Please enter your Mechanical Turk WorkerID for payment processing",
									type: "freeForm",
							},
			{
				tag: "initiative-1",
				type: "combo",
				text: "",
				likert: {   
							text: "How much do you agree with the following statement: \n\n 'The chatbot did enough to keep the conversation going'",
							options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
						},
				why: {
							options: ["the bot was responsive to me", "the bot shared information", "the bot asked the right questions",
										"the bot was unresponsive to me", "the bot gave close-ended answers", "the bot did <b> not </b> ask enough questions", 
										 "the bot asked the wrong questions"]
						}
			},			
			{
				tag: "initiative-2",
				type: "combo",
				text: "",
				likert: {	
					text: "How much do agree with the following statement: 'I had enough opportunities to drive the conversation/choose topics'",
					options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
				},
				why: {
					options: ["the bot asked enough questions about topics I chose", "the bot let me choose topics",
						 "the bot did not ask enough questions about topics I chose", "the bot chose too many topics",
							 "the bot let me drive the conversation enough", "the bot did not let me drive the conversation enough"]
				}
			},
			{
				tag: "interested",
				type: "combo",
				text: "",
				likert: {	
					text: "Did the chatbot seem interested in you?",
					options: ["1 - Not enough", "2 - The right amount", "3 - Too much"]
				},
				why: {
					options: ["the bot asked questions about me or my interests", "I felt like the bot was listening to me", "the bot followed through on the topics I raised",
					"the bot did not ask enough questions about me or my interests", "I did not feel like the bot was listening to me", "the bot gave generic responses to what I said"]
				}
			},
			{
				tag: "sharing",
				type: "combo",

				text: "",
				likert: {	
					text: "Did the chatbot share about itself?",
					options: ["1 - Not enough", "2 - The right amount", "3 - Too much"]
				},
				why: {
					options: ["the bot shared about itself", "the bot was genuine", "the bot did not share about itself", "the bot was inappropriate"]
				}
			},
			{
				tag: "fun",
				type: "combo",
				text: "",
				likert: {
					text: "How much do you agree with the following statement: 'Talking to the chatbot was fun.'",
					options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"],
				},
				why: {
					options: ["the bot was fun", "the bot had a good sense of humor", "I was interested in what the bot was saying", "the bot was boring/bland", "the bot was not engaging", "I got tired of talking to the bot"]
				}
			},
			{
				tag: "emotional-connection",
				type: "combo",
				text: "",
				likert: {
					text: "How much do you agree with the following statement: “I felt an emotional connection to the chatbot.”  For example, the chatbot understood how you were feeling, or made you feel less lonely.",
					options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
				},
				why: {
					options: ["I felt like I was talking to a friend", "the bot seemed human", "the bot made me feel less lonely",
													"the bot understood my feelings", "the bot did not like me", "the bot made me feel uncomfortable", "I did not like the bot"]
				}
			},
			{
				tag: "communication-skills",
				type: "combo",
				text: "",
				likert: {
					text: "How much do you agree with the following statement: “The chatbot displayed good communication skills” For example, the bot was coherent, did not repeat itself, understood what I was saying, and was logically correct.",
					options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
				},
				why: {
					options: ["I understood everything the bot was talking about", "what the bot said was always consistent with the scenario", "I could not understand the bot's language", "the bot repeated itself", "the bot said things that were inconsistent with the scenario", "the bot contradicted itself", "I understood what the bot was saying, but it was absurd/ridiculous", "there were times when the bot did not make sense"]
				}
			},
			{
				tag: "social-skills",
				type: "combo",
				text: "",
				likert: {
					text: "How much do you agree with the following statement: “The chatbot responded in a way that was socially appropriate” For example, the bot was polite or the bot understood social situations.",
					options: ["1 - Strongly disagree", "2 - Somewhat disagree", "3 - Neutral", "4 - Somewhat agree", "5 - Strongly agree"]
				},
				why: {
					options: ["the bot was polite", "the bot understood social situations", "the bot was rude", "the bot was inappropriate", "the bot was argumentative", "the bot was uncooperative", "the bot was too agreeable"]
				}
			},
			{
									tag: "feedback-bot",
									text: "Is there anything else you would like to say about the bot?",
									type: "freeForm",
							},
			{
									tag: "feedback-survey",
									text: "Is there anything else you would like to say about the survey?",
									type: "freeForm",
							}
			
		],

		};*/
		
		for (let question of questions) {
			if (question.type === "likert") {
				question.rating = "";
			} else if (question.type === "turn-binary") {
				question.selectedUtterances = [];
				question.notaUtterance = false;
			} else if (question.type === "turn-ternary"){
				question.notaUtterance = false;
				question.turnAnnotations = [];
				/* TO DO: Add initialization for new type of question */
			} else if (question.type === "combo") {
				question.rating = "";
				if (question.turn !== null && typeof(question.turn) !== 'undefined') {
					question.selectedUtterances = [];
					question.notaUtterance = false;
				}
				if (question.why !== null && typeof(question.why) !== 'undefined') {
					question.selectedReasons = [];
					question.somethingElseValue = "";
				}
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
			// var survey = this.urlParams.get("survey");
			return this.currentQuestionIdx === this.questions.length - 1;
		},
		currentQuestion: function () {
			// var survey = this.urlParams.get("survey");
			if (this.currentQuestionIdx < 0) {
				return {
					tag: "Dummy",
					text: "Dummy", 
					type: "Dummy",
					selectedUtterances: [],
					selectedReasons: [],
					somethingElseValue: "",
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
			if (this.isConversationOver && this.currentQuestionIdx > -1 && (this.currentQuestion.type === "turn-binary" ||
				(this.currentQuestion.type === "combo" && this.currentQuestion.turn !== null 
				&& typeof(this.currentQuestion.turn) !== 'undefined'))) {
				//if(!"selectedUtterances" in this.currentQuestion) {this.currentQuestion.selectedUtterances=[]}
				console.log("Toggling selected utterances");
				this.currentQuestion.selectedUtterances = _.xor(this.currentQuestion.selectedUtterances, [idx])
				if (this.currentQuestion.selectedUtterances.length > 0) {
					this.currentQuestion.notaUtterance = false;
				}
			}
		},
		toggleSelectedReason: function (option) {
			if (this.isConversationOver && this.currentQuestionIdx > -1 && this.currentQuestion.type === "combo"
				&& this.currentQuestion.why !== null && typeof(this.currentQuestion.why) !== 'undefined') {
				console.log("Toggling selected reasons");
				this.currentQuestion.selectedReasons = _.xor(this.currentQuestion.selectedReasons, [option])
			}
		},
		logSomethingElse: function (option) {
			if (this.isConversationOver && this.currentQuestionIdx > -1 && this.currentQuestion.type === "combo"
				&& this.currentQuestion.why !== null && typeof(this.currentQuestion.why) !== 'undefined') {
					if (this.currentQuestion.why.somethingElseValue !== "Something else") {
						console.log("Add why reason from text input");
						this.currentQuestion.somethingElseValue = option;
						console.log(this.currentQuestion.somethingElseValue);
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
			if (this.currentQuestion.type === 'turn-binary' || (this.currentQuestion.type === 'combo'
				&& this.currentQuestion.turn !== null && typeof(this.currentQuestion.turn) !== 'undefined')){
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
			/* this.addDatasetQuestion(response); */
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
			// survey = urlParams.get("survey");
			// console.log(survey);
			// console.log(this.questions[survey]);
			for (let question of this.questions) {
				if (question.type === "likert") {
					question.rating = "";
				} else if (question.type === "turn-binary" || (question.type === "combo" &&
					question.turn !== null && typeof(question.turn) !== 'undefined')) {
					question.selectedUtterances = [];
					question.notaUtterance = false;
				} else if (question.type === "combo") {
					question.rating = "";
					consolte.log("initialized rating");
					if (question.why !== null && typeof(question.why) !== 'undefined') {
						question.selectedReasons = [];
						question.somethingElseValue = "";
					}
				}
				else if (question.type === "turn-ternary"){
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
			// var survey = urlParams.get("survey");
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
			else if (this.currentQuestion.type === "combo") {
				if (!(this.currentQuestion.likert.options.includes(this.currentQuestion.rating))) {
                                        this.error = "Select one out of the given options";
                                        return false;
                                }
				if (this.currentQuestion.turn !== null && typeof(this.currentQuestion.turn) !== 'undefined') {
					if (this.currentQuestion.selectedUtterances.length === 0 && !this.currentQuestion.notaUtterance) {
						this.error = "Either select one or more utterances or check 'None of the utterances'";
						return false;
					}
				}
				if (this.currentQuestion.why !== null && typeof(this.currentQuestion.why) !== 'undefined') {
					if (this.currentQuestion.selectedReasons.length === 0 && this.currentQuestion.somethingElseValue === "") {
						this.error = "Select at least one reason";
						return false;
					}
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
