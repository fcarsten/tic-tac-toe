from tensorboard import program

tb = program.TensorBoard()
tb.configure(argv=["", "--logdir", "./graphs"])
tb.main()
