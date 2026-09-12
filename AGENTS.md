# FlyHoldem engineering contract

Read PROGRESS.md and the product specification before continuing. Keep work on astra/visual-first and preserve pushed history. V0 is frozen at visual-demo-v0. Use uv/locked dependencies; Python and other runtimes may be upgraded when useful and verified, as explicitly authorized by the user and the original project checkout. Keep data, environments, secrets, model checkpoints and large run logs ignored.

Run relevant tests before each coherent milestone commit; update PROGRESS.md with working commands, exact evidence, known failures, next step and resume command. Push verified milestones and tags when authorized and remote access is available. Never claim a gate passed without its registered evidence. Never select biological mapping parameters from poker profit.

Use PokerKit as the only rules engine. Neural scores must remain the fly's sole policy source. Preserve byte-level observation leakage tests and teacher-disconnected inference tests. A failed scientific gate blocks only dependent claims/experiments; publish negative results and continue independent engineering. Keep fixture/circuit/full and both learning-mode labels explicit. Do not run concurrent full-graph workers before measuring resource use.


## V10 fixed final policy exported before development

The registered run completed all 30,000 traversals, 4,151,160 counterfactual nodes and 1,580,726 terminal branches, producing 152,391 information sets. The resumed invocation took 3,262.040563 seconds, in addition to the initial 11.780402-second resource boundary; peak RSS was 284,524,544 bytes. Run storage is 149 MiB and the numeric policy is 35 MiB. These training quantities are not held-out poker results. Complete-journal, fixed population/deal sampling, numeric checkpoint, RNG, source and runtime export checks passed. Only the registered final iteration was exported.

Frozen policy SHA 762b180986a3983e8a43f4744c6d89cf11299e16b611df6aba2e8b62a4206955; training result ba31db6f19ecda2e54ed15875cf025394181c35e3d2df0c39ae0c03a69ad49a2; manifest 7b41e4ba99eea406c9ada054c1f5881c8a793a61cbd7cab39b42f36b50245d38; complete journal file b1cb28ae6c03a190ebfbfce7b4161451943f0f5604ea22399692105158107e0b; journal head 9b7e1bb3df640c37c1801014f12fb9584a16c8071809349c726fa0a399b47659. Training source remains f1c78c77620a834cd968672f224447e0ac910b0c95d3f36f645193cd4f490dff from code snapshot c569620; its recorded launch commit c745be2 contains the subsequent execution-identity documentation.

Proceed with the already frozen guarded evaluator: `PYTHONPATH="$PWD/runs/teacher-evaluation-runtime-v10/src" .venv/bin/python -m flyholdem.cli teacher evaluate --policy runs/teacher-external-regret-v10-policy --config runs/teacher-evaluation-runtime-v10/configs/teacher_evaluation.yaml --profile development --output runs/teacher-external-regret-v10-development`. Resume replaces --output with --resume naming that same run. Independently verify through the same snapshot with teacher verify-evaluation --run runs/teacher-external-regret-v10-development --policy runs/teacher-external-regret-v10-policy --allow-development. No held-out result exists at this checkpoint. Full confirmation remains unused until development passes and its artifacts reverify. Full Gate 2A and Gates 3–6 remain pending.

Latest activity-recording milestone e3a7866 passed CI 34705921821, including fixture/data/oracle/live/replay/avatar/human-browser checks. The frozen native viewer and environment remain unchanged. Generated models, logs and reports stay local.
