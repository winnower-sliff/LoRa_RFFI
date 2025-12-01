def calculate_flops_and_params(model, triplet_data):
    try:
        from thop import profile

        dummy_a = triplet_data[0][0:1]
        dummy_p = triplet_data[1][0:1]
        dummy_n = triplet_data[2][0:1]

        # 针对TripletNet的特殊处理
        if hasattr(model, 'embedding_net'):
            print("TripletNet detected. Calculating FLOPs...")
            embedding_net = model.embedding_net
            flops_single, params = profile(embedding_net, inputs=(dummy_a,), verbose=False)
            total_flops = flops_single
            print(f"FLOPs: {total_flops / 1e6:.6f}M")
            print(f"Params: {params / 1e3:.2f}K")
        else:
            total_flops, params = profile(model, inputs=(dummy_a, dummy_p, dummy_n), verbose=False)
            print(f"FLOPs: {total_flops / 1e6:.6f}M")
            print(f"Params: {params / 1e3:.2f}K")
    except ImportError:
        print("THOP not installed. Please install it using 'pip install thop'")
    except Exception as e:
        print(f"FLOPs计算失败: {e}")