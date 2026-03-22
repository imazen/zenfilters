//! Integration tests for zenode node definitions.

#[cfg(feature = "zenode")]
mod tests {
    use zenfilters::zenode_defs::*;
    use zenode::*;

    #[test]
    fn exposure_node_schema_matches() {
        let schema = EXPOSURE_NODE.schema();
        assert_eq!(schema.id, "zenfilters.exposure");
        assert_eq!(schema.label, "Exposure");
        assert_eq!(schema.group, NodeGroup::Tone);
        assert_eq!(schema.phase, Phase::DisplayAdjust);
        assert_eq!(schema.params.len(), 1);
        assert_eq!(schema.params[0].name, "stops");
        match &schema.params[0].kind {
            ParamKind::Float {
                min,
                max,
                default,
                identity,
                step,
            } => {
                assert_eq!(*min, -5.0);
                assert_eq!(*max, 5.0);
                assert_eq!(*default, 0.0);
                assert_eq!(*identity, 0.0);
                assert_eq!(*step, 0.1);
            }
            other => panic!("expected Float, got {other:?}"),
        }
        assert_eq!(schema.params[0].unit, "EV");
        assert_eq!(schema.params[0].slider, SliderMapping::Linear);
    }

    #[test]
    fn contrast_node_schema_matches() {
        let schema = CONTRAST_NODE.schema();
        assert_eq!(schema.id, "zenfilters.contrast");
        assert_eq!(schema.group, NodeGroup::Tone);
        assert_eq!(schema.params.len(), 1);
        assert_eq!(schema.params[0].name, "amount");
        assert_eq!(schema.params[0].slider, SliderMapping::SquareFromSlider);
    }

    #[test]
    fn saturation_node_defaults() {
        let node = Saturation::default();
        assert_eq!(node.factor, 1.0);
        let schema = SATURATION_NODE.schema();
        assert_eq!(schema.id, "zenfilters.saturation");
        assert_eq!(schema.group, NodeGroup::Color);
        assert_eq!(schema.params[0].slider, SliderMapping::FactorCentered);
    }

    #[test]
    fn clarity_node_is_neighborhood() {
        let schema = CLARITY_NODE.schema();
        assert_eq!(schema.id, "zenfilters.clarity");
        assert_eq!(schema.group, NodeGroup::Detail);
        assert_eq!(schema.phase, Phase::PreResize);
        assert!(schema.format.is_neighborhood);
        assert_eq!(schema.params.len(), 2);
    }

    #[test]
    fn vignette_node_post_resize() {
        let schema = VIGNETTE_NODE.schema();
        assert_eq!(schema.id, "zenfilters.vignette");
        assert_eq!(schema.group, NodeGroup::Effects);
        assert_eq!(schema.phase, Phase::PostResize);
        assert_eq!(schema.params.len(), 4);
    }

    #[test]
    fn dt_sigmoid_is_tonemap() {
        let schema = DT_SIGMOID_NODE.schema();
        assert_eq!(schema.id, "zenfilters.dt_sigmoid");
        assert_eq!(schema.group, NodeGroup::ToneMap);
        assert_eq!(schema.phase, Phase::ToneMap);
    }

    #[test]
    fn coalesce_groups_correct() {
        let fused = [
            EXPOSURE_NODE.schema(),
            CONTRAST_NODE.schema(),
            BLACK_POINT_NODE.schema(),
            WHITE_POINT_NODE.schema(),
            SATURATION_NODE.schema(),
            VIBRANCE_NODE.schema(),
            TEMPERATURE_NODE.schema(),
            TINT_NODE.schema(),
            DEHAZE_NODE.schema(),
        ];
        for schema in &fused {
            assert!(
                schema.coalesce.is_some(),
                "{} should have coalesce info",
                schema.id
            );
            assert_eq!(
                schema.coalesce.as_ref().unwrap().group,
                "fused_adjust",
                "{} coalesce group mismatch",
                schema.id
            );
        }
    }

    #[test]
    fn register_all_populates_registry() {
        let mut registry = NodeRegistry::new();
        register_all(&mut registry);
        assert!(
            registry.all().len() >= 32,
            "expected at least 32 nodes, got {}",
            registry.all().len()
        );
        assert!(registry.get("zenfilters.exposure").is_some());
        assert!(registry.get("zenfilters.invert").is_some());
        assert!(registry.get("zenfilters.vignette").is_some());
    }

    #[test]
    fn node_instance_get_set() {
        use zenode::traits::NodeInstance;
        let mut node = Exposure { stops: 1.5 };
        assert_eq!(node.get_param("stops"), Some(ParamValue::F32(1.5)));
        assert!(node.set_param("stops", ParamValue::F32(-2.0)));
        assert_eq!(node.stops, -2.0);
        assert!(!node.set_param("nonexistent", ParamValue::F32(0.0)));
    }

    #[test]
    fn node_instance_to_params() {
        use zenode::traits::NodeInstance;
        let node = Vibrance {
            amount: 0.3,
            protection: 1.5,
        };
        let params = node.to_params();
        assert_eq!(params.get("amount"), Some(&ParamValue::F32(0.3)));
        assert_eq!(params.get("protection"), Some(&ParamValue::F32(1.5)));
    }

    #[test]
    fn all_groups_represented() {
        let mut registry = NodeRegistry::new();
        register_all(&mut registry);

        let has = |g: NodeGroup| !registry.by_group(g).is_empty();
        assert!(has(NodeGroup::Tone), "Tone");
        assert!(has(NodeGroup::ToneRange), "ToneRange");
        assert!(has(NodeGroup::ToneMap), "ToneMap");
        assert!(has(NodeGroup::Color), "Color");
        assert!(has(NodeGroup::Detail), "Detail");
        assert!(has(NodeGroup::Effects), "Effects");
    }

    #[test]
    fn adaptive_sharpen_params_match() {
        let schema = ADAPTIVE_SHARPEN_NODE.schema();
        assert_eq!(schema.id, "zenfilters.adaptive_sharpen");
        assert_eq!(schema.params.len(), 5);
        let names: Vec<&str> = schema.params.iter().map(|p| p.name).collect();
        assert_eq!(
            names,
            &["amount", "sigma", "detail", "masking", "noise_floor"]
        );
    }

    #[test]
    fn noise_reduction_has_scales_param() {
        let schema = NOISE_REDUCTION_NODE.schema();
        let scales = schema.params.iter().find(|p| p.name == "scales").unwrap();
        match &scales.kind {
            ParamKind::Int { min, max, default } => {
                assert_eq!(*min, 1);
                assert_eq!(*max, 6);
                assert_eq!(*default, 4);
            }
            other => panic!("expected Int, got {other:?}"),
        }
    }

    #[test]
    fn color_grading_has_all_params() {
        let schema = COLOR_GRADING_NODE.schema();
        assert_eq!(schema.params.len(), 7);
        let names: Vec<&str> = schema.params.iter().map(|p| p.name).collect();
        assert!(names.contains(&"shadow_a"));
        assert!(names.contains(&"highlight_b"));
        assert!(names.contains(&"balance"));
    }

    #[test]
    fn camera_calibration_params() {
        let schema = CAMERA_CALIBRATION_NODE.schema();
        assert_eq!(schema.params.len(), 7);
        let red_sat = schema
            .params
            .iter()
            .find(|p| p.name == "red_saturation")
            .unwrap();
        match &red_sat.kind {
            ParamKind::Float {
                default, identity, ..
            } => {
                assert_eq!(*default, 1.0);
                assert_eq!(*identity, 1.0);
            }
            other => panic!("expected Float, got {other:?}"),
        }
    }

    #[test]
    fn grain_seed_is_int() {
        let schema = GRAIN_NODE.schema();
        let seed = schema.params.iter().find(|p| p.name == "seed").unwrap();
        match &seed.kind {
            ParamKind::Int { .. } => {}
            other => panic!("expected Int, got {other:?}"),
        }
    }

    #[test]
    fn create_from_registry() {
        let mut registry = NodeRegistry::new();
        register_all(&mut registry);
        let mut params = ParamMap::new();
        params.insert("stops".into(), ParamValue::F32(2.5));
        let instance = registry.create("zenfilters.exposure", &params).unwrap();
        assert_eq!(instance.get_param("stops"), Some(ParamValue::F32(2.5)));
    }
}
