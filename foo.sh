
gh api \
	--method POST \
	"/repos/maybeetree/PyPO-fork/pages" \
	-f 'build_type=workflow' \
	|| die "failed to enable pages."

