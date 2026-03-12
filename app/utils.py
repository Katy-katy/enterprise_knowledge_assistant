def stats(r):
    ns = 1e9
    
    gen_tps = r["eval_count"] / (r["eval_duration"]/ns)
    prompt_tps = r["prompt_eval_count"] / (r["prompt_eval_duration"]/ns)
    
    print("Generation speed:", round(gen_tps,2), "tok/s")
    print("Prompt speed:", round(prompt_tps,2), "tok/s")
    print("Total latency:", round(r["total_duration"]/ns,2),"s")
