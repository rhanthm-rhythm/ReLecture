#!/bin/bash
# Case Study Runner for ReLecture Paper
# This script runs the full pipeline for case study videos

set -e

VIDEO_DIR="data/videos"
OUTPUT_BASE="case_study/results"
VIDEOS=("43590" "65781" "68145")
BACKENDS=("qwen3")  # Add "chatterbox" "cosyvoice" later
STRATEGIES=("independent" "two_pass")

echo "=========================================="
echo "ReLecture Case Study Pipeline Runner"
echo "=========================================="
echo ""

# Function to run a single configuration
run_config() {
    local video_id=$1
    local backend=$2
    local strategy=$3
    local profile=$4
    local output_dir=$5

    echo "Running: video=$video_id backend=$backend strategy=$strategy profile=$profile"

    local video_path="$VIDEO_DIR/${video_id}.mp4"
    local project_file="$output_dir/project.json"

    # Determine profile settings
    local background="none"
    local accessibility="none"
    local audience="general audience"
    local describe_visuals=""

    if [ "$profile" = "cs_background" ]; then
        background="cs_background"
        audience="students without a CS background"
    elif [ "$profile" = "visual_impairment" ]; then
        accessibility="visual_impairment"
        audience="visually impaired students"
        describe_visuals="--describe-visuals"
    fi

    # Load backend config
    local backend_config="case_study/configs/${backend}.json"
    local speed=$(jq -r '.speed // 1.0' "$backend_config")
    local temperature=$(jq -r '.temperature // 0.9' "$backend_config")
    local top_p=$(jq -r '.top_p // 1.0' "$backend_config")
    local repetition_penalty=$(jq -r '.repetition_penalty // 1.05' "$backend_config")

    # Run full pipeline
    uv run python -m relecture run \
        --source-video "$video_path" \
        --project-name "case_study_${video_id}_${backend}_${strategy}_${profile}" \
        --background "$background" \
        --accessibility "$accessibility" \
        --target-audience "$audience" \
        $describe_visuals \
        --strategy "$strategy" \
        --mode full \
        --backend "$backend" \
        --speed "$speed" \
        --temperature "$temperature" \
        --top-p "$top_p" \
        --repetition-penalty "$repetition_penalty" \
        --output-filename "lecture_final.mp4"

    echo "Completed: $video_id $backend $strategy $profile"
    echo ""
}

# Main execution
for video_id in "${VIDEOS[@]}"; do
    for backend in "${BACKENDS[@]}"; do
        for strategy in "${STRATEGIES[@]}"; do
            for profile in "cs_background" "visual_impairment"; do
                output_dir="$OUTPUT_BASE/${backend}/${strategy}/${profile}/${video_id}"
                mkdir -p "$output_dir"

                # Run the configuration
                run_config "$video_id" "$backend" "$strategy" "$profile" "$output_dir" \
                    2>&1 | tee "$output_dir/run.log"

                # Copy important outputs to case_study directory
                if [ -f "output/case_study_${video_id}_${backend}_${strategy}_${profile}/project.json" ]; then
                    cp -r "output/case_study_${video_id}_${backend}_${strategy}_${profile}"/* "$output_dir/"
                fi
            done
        done
    done
done

echo "=========================================="
echo "Case study pipeline complete!"
echo "Results in: $OUTPUT_BASE"
echo "=========================================="
